"""
FBX file loader using pyassimp.

Extracts mesh data, skeleton hierarchy, skinning weights, and materials
from FBX files exported by Mixamo.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

try:
    import pyassimp
    import pyassimp.postprocess as pp
except ImportError:
    pyassimp = None

logger = logging.getLogger(__name__)


@dataclass
class BoneInfo:
    """Information about a single bone in the skeleton."""
    name: str
    parent_index: int  # -1 for root
    children_indices: list[int] = field(default_factory=list)
    # Bind-pose offset matrix (bone space -> mesh space), 4x4
    offset_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    # Local transform relative to parent, 4x4
    local_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))
    # Global (world) transform, 4x4
    global_transform: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))


@dataclass
class MeshData:
    """Extracted mesh data."""
    name: str
    positions: np.ndarray       # (N, 3) float32
    normals: np.ndarray         # (N, 3) float32
    texcoords: np.ndarray       # (N, 2) float32 or None
    indices: np.ndarray         # (M,) uint32 triangle indices
    joint_indices: np.ndarray   # (N, 4) uint16 - bone indices per vertex
    joint_weights: np.ndarray   # (N, 4) float32 - bone weights per vertex
    material_index: int = 0


@dataclass
class MaterialData:
    """Extracted material data."""
    name: str
    diffuse_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    diffuse_texture_path: str | None = None
    normal_texture_path: str | None = None
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metallic: float = 0.0
    roughness: float = 0.9


@dataclass
class FBXData:
    """All data extracted from an FBX file."""
    meshes: list[MeshData]
    bones: list[BoneInfo]
    bone_name_to_index: dict[str, int]
    materials: list[MaterialData]


def _build_node_map(node, parent_name=None, result=None):
    """Recursively build a map of node names to their parent names and transforms."""
    if result is None:
        result = {}
    name = node.name
    transform = np.array(node.transformation, dtype=np.float32).T  # assimp is row-major
    result[name] = {
        "parent": parent_name,
        "transform": transform,
        "children": [ch.name for ch in node.children],
    }
    for child in node.children:
        _build_node_map(child, name, result)
    return result


def load_fbx(filepath: str, callback=None) -> FBXData:
    """
    Load an FBX file and extract all data needed for VRM conversion.

    Args:
        filepath: Path to the FBX file.
        callback: Optional progress callback(message: str, progress: float 0-1).

    Returns:
        FBXData with meshes, bones, and materials.
    """
    if pyassimp is None:
        raise ImportError(
            "pyassimp is not installed. Install it with: pip install pyassimp\n"
            "Also ensure the Assimp shared library is available on your system."
        )

    def _progress(msg, pct):
        logger.info(msg)
        if callback:
            callback(msg, pct)

    _progress("Loading FBX file...", 0.0)

    flags = (
        pp.aiProcess_Triangulate
        | pp.aiProcess_GenNormals
        | pp.aiProcess_JoinIdenticalVertices
        | pp.aiProcess_LimitBoneWeights  # max 4 weights per vertex
        | pp.aiProcess_FlipUVs
    )

    scene = pyassimp.load(filepath, processing=flags)

    try:
        _progress("Building node hierarchy...", 0.1)
        node_map = _build_node_map(scene.rootnode)

        # --- Extract bones from all meshes ---
        _progress("Extracting skeleton...", 0.2)
        bone_name_set = set()
        bone_offset_matrices = {}

        for mesh in scene.meshes:
            for bone in mesh.bones:
                bone_name_set.add(bone.name)
                # offset_matrix: transforms from mesh space to bone space
                bone_offset_matrices[bone.name] = np.array(
                    bone.offsetmatrix, dtype=np.float32
                ).T  # transpose from row-major

        # Build ordered bone list by walking the node hierarchy
        bone_list: list[BoneInfo] = []
        bone_name_to_idx: dict[str, int] = {}

        # Find all nodes that are bones (referenced by mesh skinning)
        # Also include their ancestors up to root to preserve hierarchy
        all_bone_nodes = set(bone_name_set)
        for bname in bone_name_set:
            parent = node_map.get(bname, {}).get("parent")
            while parent and parent in node_map:
                all_bone_nodes.add(parent)
                parent = node_map[parent].get("parent")

        # Walk the tree in depth-first order to build bone list
        def _walk_bones(node_name, parent_idx=-1):
            if node_name not in node_map:
                return
            info = node_map[node_name]

            # Only add nodes that are bones or ancestors of bones
            if node_name not in all_bone_nodes:
                return

            idx = len(bone_list)
            bone_name_to_idx[node_name] = idx

            offset = bone_offset_matrices.get(
                node_name, np.eye(4, dtype=np.float32)
            )
            local_transform = info["transform"]

            bone = BoneInfo(
                name=node_name,
                parent_index=parent_idx,
                offset_matrix=offset,
                local_transform=local_transform,
            )
            bone_list.append(bone)

            if parent_idx >= 0:
                bone_list[parent_idx].children_indices.append(idx)

            for child_name in info.get("children", []):
                _walk_bones(child_name, idx)

        _walk_bones(scene.rootnode.name, -1)

        # Compute global transforms
        for bone in bone_list:
            if bone.parent_index < 0:
                bone.global_transform = bone.local_transform.copy()
            else:
                parent = bone_list[bone.parent_index]
                bone.global_transform = parent.global_transform @ bone.local_transform

        # --- Extract materials ---
        _progress("Extracting materials...", 0.4)
        materials: list[MaterialData] = []
        for mat in scene.materials:
            props = dict(mat.properties.items()) if hasattr(mat.properties, 'items') else {}
            name = props.get("name", f"Material_{len(materials)}")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            diffuse = (0.8, 0.8, 0.8, 1.0)
            diffuse_tex = None
            normal_tex = None

            # Extract color
            if hasattr(mat, 'properties'):
                for key, val in (props.items() if isinstance(props, dict) else []):
                    if 'diffuse' in str(key).lower() and hasattr(val, '__len__') and len(val) >= 3:
                        diffuse = tuple(float(v) for v in val[:4]) if len(val) >= 4 else (*[float(v) for v in val[:3]], 1.0)

            # Extract texture paths
            if hasattr(mat, 'properties'):
                for key, val in (props.items() if isinstance(props, dict) else []):
                    key_str = str(key).lower()
                    if 'file' in key_str and 'diffuse' in key_str and isinstance(val, (str, bytes)):
                        diffuse_tex = val.decode("utf-8") if isinstance(val, bytes) else val
                    elif 'file' in key_str and 'normal' in key_str and isinstance(val, (str, bytes)):
                        normal_tex = val.decode("utf-8") if isinstance(val, bytes) else val

            materials.append(MaterialData(
                name=name,
                diffuse_color=diffuse,
                diffuse_texture_path=diffuse_tex,
                normal_texture_path=normal_tex,
            ))

        if not materials:
            materials.append(MaterialData(name="DefaultMaterial"))

        # --- Extract meshes ---
        _progress("Extracting meshes...", 0.5)
        mesh_list: list[MeshData] = []

        for mi, mesh in enumerate(scene.meshes):
            _progress(f"Processing mesh {mi + 1}/{len(scene.meshes)}...",
                      0.5 + 0.4 * (mi / max(len(scene.meshes), 1)))

            n_verts = len(mesh.vertices)
            positions = np.array(mesh.vertices, dtype=np.float32).reshape(-1, 3)
            normals = np.array(mesh.normals, dtype=np.float32).reshape(-1, 3)

            # Texture coordinates
            texcoords = None
            if mesh.texturecoords is not None and len(mesh.texturecoords) > 0:
                tc = np.array(mesh.texturecoords[0], dtype=np.float32)
                if tc.ndim == 2:
                    texcoords = tc[:, :2]

            if texcoords is None:
                texcoords = np.zeros((n_verts, 2), dtype=np.float32)

            # Indices
            faces = np.array(mesh.faces, dtype=np.uint32)
            if faces.ndim == 2:
                indices = faces.reshape(-1)
            else:
                # Variable-length faces (should not happen with triangulation)
                idx_list = []
                for f in faces:
                    idx_list.extend(f)
                indices = np.array(idx_list, dtype=np.uint32)

            # Skinning weights (up to 4 per vertex)
            joint_indices = np.zeros((n_verts, 4), dtype=np.uint16)
            joint_weights = np.zeros((n_verts, 4), dtype=np.float32)

            # Collect per-vertex bone data
            vert_bone_data: list[list[tuple[int, float]]] = [[] for _ in range(n_verts)]

            for bone in mesh.bones:
                if bone.name not in bone_name_to_idx:
                    continue
                bidx = bone_name_to_idx[bone.name]
                for weight in bone.weights:
                    vid = int(weight.vertexid)
                    w = float(weight.weight)
                    if vid < n_verts and w > 0:
                        vert_bone_data[vid].append((bidx, w))

            # Sort by weight and take top 4
            for vid in range(n_verts):
                entries = sorted(vert_bone_data[vid], key=lambda x: -x[1])[:4]
                if entries:
                    total = sum(e[1] for e in entries)
                    for j, (bidx, w) in enumerate(entries):
                        joint_indices[vid, j] = bidx
                        joint_weights[vid, j] = w / total if total > 0 else 0

            mesh_list.append(MeshData(
                name=mesh.name if mesh.name else f"Mesh_{mi}",
                positions=positions,
                normals=normals,
                texcoords=texcoords,
                indices=indices,
                joint_indices=joint_indices,
                joint_weights=joint_weights,
                material_index=mesh.materialindex,
            ))

        _progress("FBX loading complete.", 1.0)

        return FBXData(
            meshes=mesh_list,
            bones=bone_list,
            bone_name_to_index=bone_name_to_idx,
            materials=materials,
        )
    finally:
        pyassimp.release(scene)
