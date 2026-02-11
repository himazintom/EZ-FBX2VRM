"""
FBX file loader using pyassimp.

Extracts mesh data, skeleton hierarchy, skinning weights, and materials
from FBX files exported by Mixamo.
"""

import ctypes
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
    embedded_textures: dict[str, bytes] = field(default_factory=dict)  # "*0" -> raw bytes


def _build_node_map(node, parent_name=None, result=None):
    """Recursively build a map of node names to their parent names and transforms."""
    if result is None:
        result = {}
    name = node.name
    transform = np.array(node.transformation, dtype=np.float32)  # pyassimp already column-major
    result[name] = {
        "parent": parent_name,
        "transform": transform,
        "children": [ch.name for ch in node.children],
    }
    for child in node.children:
        _build_node_map(child, name, result)
    return result


def _find_mesh_node_scale(node):
    """Walk the scene tree and return the scale factor of the first mesh node.

    Returns the magnitude of the first column of the mesh node's transform,
    which represents the uniform scale applied by Assimp's FBX importer to
    mesh vertices.  Returns None if no scaled mesh node is found.
    """
    if hasattr(node, 'meshes') and len(node.meshes) > 0:
        xform = np.array(node.transformation, dtype=np.float32)
        scale = float(np.linalg.norm(xform[:3, 0]))
        if scale > 1.5:  # Only return if significantly scaled
            return scale
    for child in node.children:
        result = _find_mesh_node_scale(child)
        if result is not None:
            return result
    return None


def _fix_bone_unit_scale(scene, bone_list, log):
    """Detect and fix unit scale mismatch between mesh vertices and bone transforms.

    Assimp bakes the mesh node's transform (including scale) into vertex
    positions, but bone node transforms stay in the original FBX units.
    Extract the exact scale factor from the mesh node's transform matrix
    and apply its inverse to bone translations.
    """
    if not bone_list:
        return

    mesh_node_scale = _find_mesh_node_scale(scene.rootnode)
    if mesh_node_scale is None:
        return

    scale_factor = 1.0 / mesh_node_scale
    log.info(
        f"Mesh node scale={mesh_node_scale:.4f}, "
        f"applying bone scale={scale_factor:.6f}"
    )

    # Scale translation components of every bone's local_transform
    for bone in bone_list:
        bone.local_transform[:3, 3] *= scale_factor


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

    try:
        scene_ctx = pyassimp.load(filepath, processing=flags)
    except OSError as e:
        err_msg = str(e).lower()
        if "assimp" in err_msg or "dll" in err_msg or "shared" in err_msg or "library" in err_msg:
            raise RuntimeError(
                "Assimp shared library not found. On Windows, install the Assimp DLL "
                "or run: pip install pyassimp and ensure assimp.dll is on your PATH.\n"
                f"Original error: {e}"
            ) from e
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load FBX file: {filepath}\n{e}") from e

    with scene_ctx as scene:
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
                )  # pyassimp already column-major

        # Build ordered bone list by walking the node hierarchy.
        # Collapse $AssimpFbx$ intermediate nodes (Translation/PreRotation/Rotation)
        # by folding their transforms into the next real bone node.
        bone_list: list[BoneInfo] = []
        bone_name_to_idx: dict[str, int] = {}

        # Determine which nodes are "real" (actual bones or RootNode)
        # vs intermediate ($AssimpFbx$ helper nodes created by Assimp)
        def _is_intermediate(name):
            return '$AssimpFbx$' in name

        # Find all real nodes that should be in the skeleton:
        # actual bone nodes + their non-intermediate ancestors
        real_bone_nodes = set(bone_name_set)
        for bname in bone_name_set:
            parent = node_map.get(bname, {}).get("parent")
            while parent and parent in node_map:
                if not _is_intermediate(parent):
                    real_bone_nodes.add(parent)
                parent = node_map[parent].get("parent")

        def _walk_bones(node_name, parent_idx=-1, accumulated_transform=None):
            if node_name not in node_map:
                return
            info = node_map[node_name]
            local_xform = info["transform"]

            # Accumulate transform through intermediate nodes
            if accumulated_transform is not None:
                local_xform = accumulated_transform @ local_xform

            if _is_intermediate(node_name):
                # Skip this node, pass accumulated transform to children
                for child_name in info.get("children", []):
                    _walk_bones(child_name, parent_idx, local_xform)
                return

            # Not in skeleton? Skip but recurse children (no accumulation)
            if node_name not in real_bone_nodes:
                return

            idx = len(bone_list)
            bone_name_to_idx[node_name] = idx

            offset = bone_offset_matrices.get(
                node_name, np.eye(4, dtype=np.float32)
            )

            bone = BoneInfo(
                name=node_name,
                parent_index=parent_idx,
                offset_matrix=offset,
                local_transform=local_xform,
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

        # --- Detect and fix unit scale mismatch ---
        # Assimp bakes mesh-node transforms (which may include scale) into
        # vertex positions, but bone transforms stay in original FBX units.
        # Detect this by comparing mesh extents to bone extents.
        _progress("Checking unit scale...", 0.35)
        _fix_bone_unit_scale(scene, bone_list, logger)

        # Recompute global transforms after scale fix
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
                logger.debug(f"Mesh '{mesh.name or mi}' has no texture coordinates, using zeros")
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
                        joint_weights[vid, j] = w / total if total > 1e-8 else 0.0

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

        # --- Extract embedded textures ---
        embedded_textures: dict[str, bytes] = {}
        for ti in range(len(scene.textures)):
            tex_ptr = scene.textures[ti]
            t = tex_ptr.contents if hasattr(tex_ptr, 'contents') else tex_ptr
            # Compressed textures: mHeight==0, mWidth==byte_count
            if t.mHeight == 0 and t.mWidth > 0:
                try:
                    data_ptr = t.pcData
                    if data_ptr:
                        byte_ptr = ctypes.cast(
                            data_ptr, ctypes.POINTER(ctypes.c_ubyte * t.mWidth)
                        )
                        raw = bytes(byte_ptr.contents)
                        embedded_textures[f"*{ti}"] = raw
                        logger.info(f"Extracted embedded texture *{ti}: {len(raw)} bytes")
                except Exception as e:
                    logger.warning(f"Failed to extract embedded texture *{ti}: {e}")

        _progress("FBX loading complete.", 1.0)

        return FBXData(
            meshes=mesh_list,
            bones=bone_list,
            bone_name_to_index=bone_name_to_idx,
            materials=materials,
            embedded_textures=embedded_textures,
        )
