"""
VRM 0.x file builder.

Constructs a glTF 2.0 binary (GLB) file with VRM extensions
from extracted FBX data.
"""

import io
import json
import logging
import struct
from pathlib import Path

import numpy as np

try:
    from pygltflib import (
        GLTF2,
        Accessor,
        Asset,
        Attributes,
        Buffer,
        BufferView,
        Image,
        Material,
        Mesh,
        Node,
        Primitive,
        Scene,
        Skin,
        Texture,
        TextureInfo,
    )
    from pygltflib import (
        FLOAT as GLTF_FLOAT,
        UNSIGNED_INT as GLTF_UINT,
        UNSIGNED_SHORT as GLTF_USHORT,
        UNSIGNED_BYTE as GLTF_UBYTE,
        SCALAR,
        VEC2,
        VEC3,
        VEC4,
        MAT4,
        ARRAY_BUFFER,
        ELEMENT_ARRAY_BUFFER,
    )
except ImportError:
    GLTF2 = None

from .fbx_loader import FBXData, MeshData, BoneInfo
from .bone_mapping import (
    build_bone_mapping,
    validate_bone_mapping,
    VRM_REQUIRED_BONES,
    VRM_OPTIONAL_BONES,
)

logger = logging.getLogger(__name__)

# VRM 0.x coordinate system uses right-handed Y-up with -Z forward (same as glTF)
# FBX models may use Z-up; detect and convert automatically.

# Z-up to Y-up rotation: -90° around X axis
# (x, y, z) -> (x, z, -y)
_Z_UP_TO_Y_UP = np.array([
    [1,  0,  0,  0],
    [0,  0,  1,  0],
    [0, -1,  0,  0],
    [0,  0,  0,  1],
], dtype=np.float32)

# Bone coordinate fix: +90° around Y axis
# Mesh vertices (local space) and bone transforms (scene root) use different
# lateral axes due to the mesh node's axis-swap transform baked by Assimp.
# This rotation aligns bone X/Z with the mesh's rotated X/Z.
# (x, y, z) -> (z, y, -x)
_BONE_AXIS_FIX = np.array([
    [ 0, 0, 1, 0],
    [ 0, 1, 0, 0],
    [-1, 0, 0, 0],
    [ 0, 0, 0, 1],
], dtype=np.float32)


def _detect_z_up(fbx_data: 'FBXData') -> bool:
    """Return True if model appears to be Z-up (Z extent >> Y extent)."""
    for mesh in fbx_data.meshes:
        if len(mesh.positions) == 0:
            continue
        extents = mesh.positions.max(axis=0) - mesh.positions.min(axis=0)
        if extents[2] > extents[1] * 1.5:
            return True
    return False


class VRMBuilder:
    """Builds a VRM 0.x file from FBX data."""

    def __init__(self, fbx_data: FBXData, fbx_path: str = ""):
        if GLTF2 is None:
            raise ImportError("pygltflib is required: pip install pygltflib")
        self.fbx = fbx_data
        self.fbx_dir = str(Path(fbx_path).parent) if fbx_path else ""
        self.gltf = GLTF2()
        self._coord_fix = None  # set during build if Z-up detected
        self.gltf.asset = Asset(version="2.0", generator="EZ-FBX2VRM")
        self.gltf.scene = 0

        # Binary buffer accumulator
        self._buffer_data = bytearray()

        # Tracking
        self._accessor_count = 0
        self._bufferview_count = 0
        self._node_count = 0
        self._mesh_count = 0
        self._material_count = 0
        self._texture_count = 0
        self._image_count = 0

        # Bone index mapping: fbx bone index -> gltf node index
        self._bone_to_node: dict[int, int] = {}
        # VRM humanoid bone entries
        self._vrm_human_bones: list[dict] = []

    def build(self, meta: dict | None = None, callback=None) -> bytes:
        """
        Build the VRM file.

        Args:
            meta: Optional VRM metadata dict (title, author, etc.)
            callback: Optional progress callback(message, progress).

        Returns:
            bytes of the complete GLB file with VRM extensions.
        """
        def _progress(msg, pct):
            logger.info(msg)
            if callback:
                callback(msg, pct)

        # Auto-detect Z-up and prepare coordinate fix
        if _detect_z_up(self.fbx):
            logger.info("Detected Z-up model, will convert to Y-up for VRM output")
            self._coord_fix = _Z_UP_TO_Y_UP
        else:
            self._coord_fix = None

        _progress("Building skeleton nodes...", 0.0)
        self._build_skeleton()

        _progress("Building materials...", 0.2)
        self._build_materials()

        _progress("Building meshes...", 0.4)
        self._build_meshes()

        _progress("Building skin...", 0.6)
        self._build_skin()

        _progress("Building scene...", 0.7)
        self._build_scene()

        _progress("Adding VRM extension...", 0.8)
        self._build_vrm_extension(meta or {})

        _progress("Encoding GLB...", 0.9)
        result = self._encode_glb()

        _progress("VRM build complete.", 1.0)
        return result

    def _add_buffer_view(self, data: bytes, target: int | None = None) -> int:
        """Add data to the buffer and create a BufferView. Returns view index."""
        # Align to 4 bytes
        while len(self._buffer_data) % 4 != 0:
            self._buffer_data.append(0)

        offset = len(self._buffer_data)
        self._buffer_data.extend(data)

        bv = BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(data),
        )
        if target is not None:
            bv.target = target

        self.gltf.bufferViews.append(bv)
        idx = self._bufferview_count
        self._bufferview_count += 1
        return idx

    def _add_accessor(
        self, buffer_view: int, component_type: int, count: int,
        type_: str, min_vals=None, max_vals=None
    ) -> int:
        """Create an Accessor. Returns accessor index."""
        acc = Accessor(
            bufferView=buffer_view,
            byteOffset=0,
            componentType=component_type,
            count=count,
            type=type_,
        )
        if min_vals is not None:
            acc.min = min_vals
        if max_vals is not None:
            acc.max = max_vals

        self.gltf.accessors.append(acc)
        idx = self._accessor_count
        self._accessor_count += 1
        return idx

    def _build_skeleton(self):
        """Create glTF nodes for each bone in the skeleton."""
        bones = self.fbx.bones
        if not bones:
            logger.warning("No bones found in FBX data")
            return

        # Create a glTF node for each bone
        for bi, bone in enumerate(bones):
            node = Node(name=bone.name)

            local_xform = bone.local_transform
            # For root bones, apply axis alignment so bones match the
            # Z_UP_TO_Y_UP-rotated mesh coordinate system
            if self._coord_fix is not None and bone.parent_index < 0:
                local_xform = _BONE_AXIS_FIX @ local_xform

            # Decompose local transform into TRS
            t, r, s = _decompose_matrix(local_xform)
            node.translation = t.tolist()
            node.rotation = r.tolist()  # [x, y, z, w]
            node.scale = s.tolist()

            self.gltf.nodes.append(node)
            node_idx = self._node_count
            self._node_count += 1
            self._bone_to_node[bi] = node_idx

        # Set children references
        for bi, bone in enumerate(bones):
            node_idx = self._bone_to_node[bi]
            child_node_indices = [self._bone_to_node[ci] for ci in bone.children_indices]
            if child_node_indices:
                self.gltf.nodes[node_idx].children = child_node_indices

        # Build VRM humanoid bone mapping
        bone_names = [b.name for b in bones]
        mapping = build_bone_mapping(bone_names)
        is_valid, missing = validate_bone_mapping(mapping)

        if not is_valid:
            logger.warning(f"Missing required VRM bones: {missing}")
            logger.warning("The output VRM may not work correctly in all applications.")

        for bone_name, vrm_name in mapping.items():
            if bone_name not in self.fbx.bone_name_to_index:
                logger.warning(f"Bone '{bone_name}' in mapping but not in FBX data, skipping")
                continue
            bi = self.fbx.bone_name_to_index[bone_name]
            node_idx = self._bone_to_node[bi]
            self._vrm_human_bones.append({
                "bone": vrm_name,
                "node": node_idx,
                "useDefaultValues": True,
            })

    def _build_materials(self):
        """Create glTF PBR materials from FBX materials."""
        for mat_data in self.fbx.materials:
            material = Material(name=mat_data.name)
            material.pbrMetallicRoughness = {
                "baseColorFactor": list(mat_data.diffuse_color),
                "metallicFactor": mat_data.metallic,
                "roughnessFactor": mat_data.roughness,
            }

            # Load diffuse texture if available
            if mat_data.diffuse_texture_path:
                tex_idx = self._load_texture(mat_data.diffuse_texture_path)
                if tex_idx is not None:
                    material.pbrMetallicRoughness["baseColorTexture"] = {
                        "index": tex_idx,
                        "texCoord": 0,
                    }
                    # When texture is present, use white factor so texture is not tinted
                    material.pbrMetallicRoughness["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]

            material.doubleSided = True
            self.gltf.materials.append(material)
            self._material_count += 1

        if self._material_count == 0:
            # Default material
            material = Material(name="DefaultMaterial")
            material.pbrMetallicRoughness = {
                "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            }
            material.doubleSided = True
            self.gltf.materials.append(material)
            self._material_count += 1

    def _load_texture(self, tex_path: str) -> int | None:
        """Load a texture file and add it to the glTF. Returns texture index or None."""
        from PIL import Image as PILImage

        try:
            # Handle embedded textures (Assimp uses "*0", "*1", etc.)
            if tex_path.startswith('*') and tex_path[1:].isdigit():
                raw_data = self.fbx.embedded_textures.get(tex_path)
                if raw_data is None:
                    logger.warning(f"Embedded texture {tex_path} not found in FBX data")
                    return None
                img = PILImage.open(io.BytesIO(raw_data))
                tex_name = f"embedded_{tex_path[1:]}"
            else:
                # External texture file
                tex_path = tex_path.replace("\\", "/")
                full_path = Path(self.fbx_dir) / tex_path if self.fbx_dir else Path(tex_path)
                if not full_path.exists():
                    full_path = Path(self.fbx_dir) / Path(tex_path).name
                    if not full_path.exists():
                        logger.warning(f"Texture not found: {tex_path}")
                        return None
                img = PILImage.open(str(full_path))
                tex_name = full_path.stem

            img = img.convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_data = buf.getvalue()

            bv_idx = self._add_buffer_view(png_data)

            gltf_image = Image(
                bufferView=bv_idx,
                mimeType="image/png",
                name=tex_name,
            )
            self.gltf.images.append(gltf_image)
            img_idx = self._image_count
            self._image_count += 1

            gltf_texture = Texture(source=img_idx)
            self.gltf.textures.append(gltf_texture)
            tex_idx = self._texture_count
            self._texture_count += 1

            logger.info(f"Loaded texture: {tex_path} ({img.size[0]}x{img.size[1]})")
            return tex_idx
        except Exception as e:
            logger.warning(f"Failed to load texture {tex_path}: {e}")
            return None

    def _build_meshes(self):
        """Create glTF meshes with skinning attributes."""
        for mesh_data in self.fbx.meshes:
            if len(mesh_data.positions) == 0 or len(mesh_data.indices) == 0:
                logger.warning(f"Skipping empty mesh: {mesh_data.name}")
                continue

            # Apply coordinate conversion if needed (Z-up -> Y-up)
            positions = mesh_data.positions.astype(np.float32)
            normals = mesh_data.normals.astype(np.float32)
            if self._coord_fix is not None:
                rot3 = self._coord_fix[:3, :3]
                positions = np.ascontiguousarray((rot3 @ positions.T).T)
                normals = np.ascontiguousarray((rot3 @ normals.T).T)

            # Position accessor
            pos_data = positions.tobytes()
            pos_bv = self._add_buffer_view(pos_data, target=ARRAY_BUFFER)
            pos_min = positions.min(axis=0).tolist()
            pos_max = positions.max(axis=0).tolist()
            pos_acc = self._add_accessor(
                pos_bv, GLTF_FLOAT, len(positions), VEC3,
                min_vals=pos_min, max_vals=pos_max
            )

            # Normal accessor
            norm_data = normals.tobytes()
            norm_bv = self._add_buffer_view(norm_data, target=ARRAY_BUFFER)
            norm_acc = self._add_accessor(
                norm_bv, GLTF_FLOAT, len(mesh_data.normals), VEC3
            )

            # Texcoord accessor
            tc_data = mesh_data.texcoords.astype(np.float32).tobytes()
            tc_bv = self._add_buffer_view(tc_data, target=ARRAY_BUFFER)
            tc_acc = self._add_accessor(
                tc_bv, GLTF_FLOAT, len(mesh_data.texcoords), VEC2
            )

            # Joint indices accessor (UNSIGNED_SHORT)
            ji_data = mesh_data.joint_indices.astype(np.uint16).tobytes()
            ji_bv = self._add_buffer_view(ji_data, target=ARRAY_BUFFER)
            ji_acc = self._add_accessor(
                ji_bv, GLTF_USHORT, len(mesh_data.joint_indices), VEC4
            )

            # Joint weights accessor
            jw_data = mesh_data.joint_weights.astype(np.float32).tobytes()
            jw_bv = self._add_buffer_view(jw_data, target=ARRAY_BUFFER)
            jw_acc = self._add_accessor(
                jw_bv, GLTF_FLOAT, len(mesh_data.joint_weights), VEC4
            )

            # Index accessor
            idx_data = mesh_data.indices.astype(np.uint32).tobytes()
            idx_bv = self._add_buffer_view(idx_data, target=ELEMENT_ARRAY_BUFFER)
            idx_acc = self._add_accessor(
                idx_bv, GLTF_UINT, len(mesh_data.indices), SCALAR
            )

            # Primitive
            attributes = Attributes(
                POSITION=pos_acc,
                NORMAL=norm_acc,
                TEXCOORD_0=tc_acc,
                JOINTS_0=ji_acc,
                WEIGHTS_0=jw_acc,
            )

            mat_idx = max(0, min(mesh_data.material_index, self._material_count - 1))

            primitive = Primitive(
                attributes=attributes,
                indices=idx_acc,
                material=mat_idx,
            )

            mesh = Mesh(
                name=mesh_data.name,
                primitives=[primitive],
            )
            self.gltf.meshes.append(mesh)

            # Create a node for this mesh
            mesh_node = Node(
                name=f"{mesh_data.name}_node",
                mesh=self._mesh_count,
            )
            self.gltf.nodes.append(mesh_node)
            self._node_count += 1
            self._mesh_count += 1

    def _build_skin(self):
        """Create glTF skin for skeletal animation."""
        if not self._bone_to_node:
            return

        bones = self.fbx.bones
        joint_nodes = [self._bone_to_node[i] for i in range(len(bones))]

        # Compute global transforms from the node hierarchy (local transforms)
        # to ensure consistency with the glTF node tree.
        # Root bones get the same axis fix applied in _build_skeleton.
        n = len(bones)
        global_xforms = np.zeros((n, 4, 4), dtype=np.float32)
        for i, bone in enumerate(bones):
            local = bone.local_transform
            if bone.parent_index < 0 and self._coord_fix is not None:
                local = _BONE_AXIS_FIX @ local
            if bone.parent_index < 0:
                global_xforms[i] = local
            else:
                global_xforms[i] = global_xforms[bone.parent_index] @ bone.local_transform

        # Inverse bind matrices = inv(global_transform) for each bone
        ibm_list = []
        for i, bone in enumerate(bones):
            try:
                ibm = np.linalg.inv(global_xforms[i])
            except np.linalg.LinAlgError:
                logger.warning(f"Singular global transform for bone '{bone.name}', using identity")
                ibm = np.eye(4, dtype=np.float32)
            ibm_list.append(ibm)

        ibm_array = np.array(ibm_list, dtype=np.float32)
        # glTF expects column-major layout for mat4: transpose each matrix
        ibm_col_major = np.ascontiguousarray(ibm_array.transpose(0, 2, 1), dtype=np.float32)
        ibm_data = ibm_col_major.tobytes()
        ibm_bv = self._add_buffer_view(ibm_data)
        ibm_acc = self._add_accessor(
            ibm_bv, GLTF_FLOAT, len(bones), MAT4
        )

        # Find the root bone node (skeleton root)
        root_idx = self._bone_to_node.get(0, 0)

        skin = Skin(
            name="Armature",
            joints=joint_nodes,
            skeleton=root_idx,
            inverseBindMatrices=ibm_acc,
        )
        self.gltf.skins.append(skin)

        # Assign skin to mesh nodes
        for node in self.gltf.nodes:
            if node.mesh is not None:
                node.skin = 0

    def _build_scene(self):
        """Build the glTF scene with root nodes."""
        root_nodes = []

        # Add root bone node(s)
        for bi, bone in enumerate(self.fbx.bones):
            if bone.parent_index < 0 and bi in self._bone_to_node:
                root_nodes.append(self._bone_to_node[bi])

        # Add mesh nodes
        for i, node in enumerate(self.gltf.nodes):
            if node.mesh is not None:
                root_nodes.append(i)

        scene = Scene(name="Scene", nodes=root_nodes)
        self.gltf.scenes.append(scene)

    def _build_vrm_extension(self, meta: dict):
        """Build and inject the VRM 0.x extension into the glTF."""
        vrm_ext = {
            "exporterVersion": "EZ-FBX2VRM-1.0",
            "specVersion": "0.0",

            "meta": {
                "title": meta.get("title", "Converted Model"),
                "version": meta.get("version", "1.0"),
                "author": meta.get("author", "Unknown"),
                "contactInformation": meta.get("contact", ""),
                "reference": meta.get("reference", ""),
                "allowedUserName": meta.get("allowedUser", "Everyone"),
                "violentUssageName": meta.get("violentUsage", "Disallow"),
                "sexualUssageName": meta.get("sexualUsage", "Disallow"),
                "commercialUssageName": meta.get("commercialUsage", "Disallow"),
                "otherPermissionUrl": "",
                "licenseName": meta.get("license", "CC0"),
                "otherLicenseUrl": "",
            },

            "humanoid": {
                "humanBones": self._vrm_human_bones,
                "armStretch": 0.05,
                "legStretch": 0.05,
                "upperArmTwist": 0.5,
                "lowerArmTwist": 0.5,
                "upperLegTwist": 0.5,
                "lowerLegTwist": 0.5,
                "feetSpacing": 0.0,
                "hasTranslationDoF": False,
            },

            "firstPerson": {
                "firstPersonBone": self._get_head_node_index(),
                "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
                "meshAnnotations": [],
                "lookAtTypeName": "Bone",
                "lookAtHorizontalInner": {
                    "curve": [0, 0, 0, 1, 1, 1, 1, 0],
                    "xRange": 90,
                    "yRange": 10,
                },
                "lookAtHorizontalOuter": {
                    "curve": [0, 0, 0, 1, 1, 1, 1, 0],
                    "xRange": 90,
                    "yRange": 10,
                },
                "lookAtVerticalDown": {
                    "curve": [0, 0, 0, 1, 1, 1, 1, 0],
                    "xRange": 90,
                    "yRange": 10,
                },
                "lookAtVerticalUp": {
                    "curve": [0, 0, 0, 1, 1, 1, 1, 0],
                    "xRange": 90,
                    "yRange": 10,
                },
            },

            "blendShapeMaster": {
                "blendShapeGroups": [],
            },

            "secondaryAnimation": {
                "boneGroups": [],
                "colliderGroups": [],
            },

            "materialProperties": self._build_vrm_materials(),
        }

        # Inject as extension
        if not hasattr(self.gltf, 'extensions') or self.gltf.extensions is None:
            self.gltf.extensions = {}
        self.gltf.extensions["VRM"] = vrm_ext

        if not self.gltf.extensionsUsed:
            self.gltf.extensionsUsed = []
        if "VRM" not in self.gltf.extensionsUsed:
            self.gltf.extensionsUsed.append("VRM")

    def _get_head_node_index(self) -> int:
        """Get the glTF node index for the head bone."""
        for entry in self._vrm_human_bones:
            if entry["bone"] == "head":
                return entry["node"]
        return 0

    def _build_vrm_materials(self) -> list[dict]:
        """Build VRM material properties list."""
        vrm_mats = []
        for i, mat in enumerate(self.gltf.materials):
            vrm_mat: dict = {
                "name": mat.name or f"Material_{i}",
                "shader": "VRM_USE_GLTFSHADER",
                "renderQueue": 2000,
                "keywordMap": {},
                "tagMap": {
                    "RenderType": "Opaque",
                },
                "floatProperties": {},
                "vectorProperties": {},
                "textureProperties": {},
            }
            vrm_mats.append(vrm_mat)
        return vrm_mats

    def _encode_glb(self) -> bytes:
        """Encode the complete glTF + buffer as GLB binary."""
        # Finalize the buffer
        # Pad buffer to 4-byte alignment
        while len(self._buffer_data) % 4 != 0:
            self._buffer_data.append(0)

        self.gltf.buffers = [Buffer(byteLength=len(self._buffer_data))]

        # Convert gltf to JSON
        gltf_dict = _gltf_to_dict(self.gltf)

        json_str = json.dumps(gltf_dict, ensure_ascii=False, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        # Pad JSON to 4-byte alignment
        while len(json_bytes) % 4 != 0:
            json_bytes += b' '

        # GLB structure:
        # Header: magic(4) + version(4) + length(4) = 12 bytes
        # JSON chunk: length(4) + type(4) + data
        # BIN chunk: length(4) + type(4) + data
        bin_data = bytes(self._buffer_data)
        while len(bin_data) % 4 != 0:
            bin_data += b'\x00'

        total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_data)

        out = io.BytesIO()
        # Header
        out.write(struct.pack('<I', 0x46546C67))  # glTF magic
        out.write(struct.pack('<I', 2))             # version 2
        out.write(struct.pack('<I', total_length))

        # JSON chunk
        out.write(struct.pack('<I', len(json_bytes)))
        out.write(struct.pack('<I', 0x4E4F534A))  # JSON
        out.write(json_bytes)

        # BIN chunk
        out.write(struct.pack('<I', len(bin_data)))
        out.write(struct.pack('<I', 0x004E4942))   # BIN\0
        out.write(bin_data)

        return out.getvalue()


def _decompose_matrix(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 transformation matrix into translation, rotation (quaternion), scale.
    Returns (translation[3], rotation[4] as xyzw, scale[3]).
    """
    translation = mat[:3, 3].copy()

    # Extract scale
    sx = np.linalg.norm(mat[:3, 0])
    sy = np.linalg.norm(mat[:3, 1])
    sz = np.linalg.norm(mat[:3, 2])
    # Clamp scales to avoid zero-division; treat near-zero scale as 1.0
    sx = sx if sx > 1e-6 else 1.0
    sy = sy if sy > 1e-6 else 1.0
    sz = sz if sz > 1e-6 else 1.0
    scale = np.array([sx, sy, sz], dtype=np.float32)

    # Extract rotation matrix (remove scale)
    rot_mat = np.zeros((3, 3), dtype=np.float64)
    rot_mat[:, 0] = mat[:3, 0] / sx
    rot_mat[:, 1] = mat[:3, 1] / sy
    rot_mat[:, 2] = mat[:3, 2] / sz

    # Convert rotation matrix to quaternion (xyzw)
    quat = _mat3_to_quaternion(rot_mat)

    return translation.astype(np.float32), quat.astype(np.float32), scale.astype(np.float32)


def _mat3_to_quaternion(m: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float64)
    # Normalize
    length = np.linalg.norm(q)
    if length > 1e-10:
        q /= length
    return q


def _gltf_to_dict(gltf: GLTF2) -> dict:
    """Convert a GLTF2 object to a JSON-serializable dict, including extensions."""
    # Use pygltflib's built-in serialization, then add extensions
    # We need to manually build the dict to properly include VRM extension

    d: dict = {
        "asset": {
            "version": gltf.asset.version,
            "generator": gltf.asset.generator,
        }
    }

    if gltf.extensionsUsed:
        d["extensionsUsed"] = gltf.extensionsUsed

    if gltf.scenes:
        d["scenes"] = []
        for scene in gltf.scenes:
            sd: dict = {}
            if scene.name:
                sd["name"] = scene.name
            if scene.nodes:
                sd["nodes"] = scene.nodes
            d["scenes"].append(sd)
        d["scene"] = gltf.scene

    if gltf.nodes:
        d["nodes"] = []
        for node in gltf.nodes:
            nd: dict = {}
            if node.name:
                nd["name"] = node.name
            if node.children:
                nd["children"] = node.children
            if node.mesh is not None:
                nd["mesh"] = node.mesh
            if node.skin is not None:
                nd["skin"] = node.skin
            if node.translation and any(v != 0 for v in node.translation):
                nd["translation"] = node.translation
            if node.rotation and not (
                node.rotation[0] == 0 and node.rotation[1] == 0 and
                node.rotation[2] == 0 and node.rotation[3] == 1
            ):
                nd["rotation"] = node.rotation
            if node.scale and not all(abs(v - 1.0) < 1e-6 for v in node.scale):
                nd["scale"] = node.scale
            d["nodes"].append(nd)

    if gltf.meshes:
        d["meshes"] = []
        for mesh in gltf.meshes:
            md: dict = {"primitives": []}
            if mesh.name:
                md["name"] = mesh.name
            for prim in mesh.primitives:
                pd: dict = {
                    "attributes": {},
                }
                attrs = prim.attributes
                if attrs.POSITION is not None:
                    pd["attributes"]["POSITION"] = attrs.POSITION
                if attrs.NORMAL is not None:
                    pd["attributes"]["NORMAL"] = attrs.NORMAL
                if attrs.TEXCOORD_0 is not None:
                    pd["attributes"]["TEXCOORD_0"] = attrs.TEXCOORD_0
                if attrs.JOINTS_0 is not None:
                    pd["attributes"]["JOINTS_0"] = attrs.JOINTS_0
                if attrs.WEIGHTS_0 is not None:
                    pd["attributes"]["WEIGHTS_0"] = attrs.WEIGHTS_0
                if prim.indices is not None:
                    pd["indices"] = prim.indices
                if prim.material is not None:
                    pd["material"] = prim.material
                md["primitives"].append(pd)
            d["meshes"].append(md)

    if gltf.accessors:
        d["accessors"] = []
        for acc in gltf.accessors:
            ad: dict = {
                "bufferView": acc.bufferView,
                "byteOffset": acc.byteOffset,
                "componentType": acc.componentType,
                "count": acc.count,
                "type": acc.type,
            }
            if acc.min is not None:
                ad["min"] = acc.min
            if acc.max is not None:
                ad["max"] = acc.max
            d["accessors"].append(ad)

    if gltf.bufferViews:
        d["bufferViews"] = []
        for bv in gltf.bufferViews:
            bvd: dict = {
                "buffer": bv.buffer,
                "byteOffset": bv.byteOffset,
                "byteLength": bv.byteLength,
            }
            if bv.target is not None:
                bvd["target"] = bv.target
            d["bufferViews"].append(bvd)

    if gltf.buffers:
        d["buffers"] = [{"byteLength": b.byteLength} for b in gltf.buffers]

    if gltf.materials:
        d["materials"] = []
        for mat in gltf.materials:
            matd: dict = {}
            if mat.name:
                matd["name"] = mat.name
            if mat.pbrMetallicRoughness:
                matd["pbrMetallicRoughness"] = mat.pbrMetallicRoughness
            if mat.doubleSided:
                matd["doubleSided"] = mat.doubleSided
            d["materials"].append(matd)

    if gltf.textures:
        d["textures"] = []
        for tex in gltf.textures:
            td: dict = {}
            if tex.source is not None:
                td["source"] = tex.source
            d["textures"].append(td)

    if gltf.images:
        d["images"] = []
        for img in gltf.images:
            imd: dict = {}
            if img.bufferView is not None:
                imd["bufferView"] = img.bufferView
            if img.mimeType:
                imd["mimeType"] = img.mimeType
            if img.name:
                imd["name"] = img.name
            d["images"].append(imd)

    if gltf.skins:
        d["skins"] = []
        for skin in gltf.skins:
            sd: dict = {
                "joints": skin.joints,
            }
            if skin.name:
                sd["name"] = skin.name
            if skin.skeleton is not None:
                sd["skeleton"] = skin.skeleton
            if skin.inverseBindMatrices is not None:
                sd["inverseBindMatrices"] = skin.inverseBindMatrices
            d["skins"].append(sd)

    # Add VRM extension
    if hasattr(gltf, 'extensions') and gltf.extensions:
        d["extensions"] = gltf.extensions

    return d
