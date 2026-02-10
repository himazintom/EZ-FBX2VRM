"""
OpenGL 3D renderer using moderngl for offscreen rendering.

Renders skinned meshes with basic Blinn-Phong lighting.
Outputs PIL Images for display in tkinter.
"""

import logging
import math

import numpy as np
from PIL import Image as PILImage

try:
    import moderngl
except ImportError:
    moderngl = None

logger = logging.getLogger(__name__)

VERTEX_SHADER = """
#version 330 core

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_bone_matrices[128];
uniform int u_use_skinning;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord;
in vec4 in_joints;
in vec4 in_weights;

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_texcoord;

void main() {
    vec4 pos = vec4(in_position, 1.0);
    vec4 norm = vec4(in_normal, 0.0);

    if (u_use_skinning == 1) {
        ivec4 ji = ivec4(in_joints);
        mat4 skin =
            in_weights.x * u_bone_matrices[ji.x] +
            in_weights.y * u_bone_matrices[ji.y] +
            in_weights.z * u_bone_matrices[ji.z] +
            in_weights.w * u_bone_matrices[ji.w];
        pos = skin * pos;
        norm = skin * norm;
    }

    vec4 world = u_model * pos;
    v_world_pos = world.xyz;
    v_normal = normalize(mat3(u_model) * norm.xyz);
    v_texcoord = in_texcoord;
    gl_Position = u_projection * u_view * world;
}
"""

FRAGMENT_SHADER = """
#version 330 core

uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform vec3 u_ambient;
uniform vec4 u_base_color;
uniform sampler2D u_texture;
uniform int u_use_texture;
uniform vec3 u_camera_pos;

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_texcoord;

out vec4 f_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_camera_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.3;

    vec4 albedo = u_base_color;
    if (u_use_texture == 1) {
        albedo *= texture(u_texture, v_texcoord);
    }

    vec3 color = (u_ambient + u_light_color * diff) * albedo.rgb + vec3(spec);
    f_color = vec4(color, albedo.a);
}
"""

GRID_VS = """
#version 330 core
uniform mat4 u_vp;
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = u_vp * vec4(in_position, 1.0);
}
"""

GRID_FS = """
#version 330 core
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
"""


class OrbitCamera:
    """Orbit camera with left-drag rotate, right-drag zoom, shift+left-drag pan."""

    def __init__(self):
        self.target = np.array([0.0, 0.85, 0.0], dtype=np.float32)
        self.distance = 3.0
        self.yaw = 0.0       # degrees, horizontal
        self.pitch = 15.0     # degrees, vertical
        self.fov = 45.0
        self.near = 0.01
        self.far = 100.0

    @property
    def eye(self) -> np.ndarray:
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        x = self.distance * math.cos(pitch_r) * math.sin(yaw_r)
        y = self.distance * math.sin(pitch_r)
        z = self.distance * math.cos(pitch_r) * math.cos(yaw_r)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float):
        """Left-click drag: rotate."""
        self.yaw += dx * 0.5
        self.pitch = max(-89, min(89, self.pitch + dy * 0.5))

    def zoom(self, dx: float, dy: float):
        """Right-click drag: zoom (vertical motion)."""
        factor = 1.0 + dy * 0.005
        self.distance = max(0.1, min(50.0, self.distance * factor))

    def pan(self, dx: float, dy: float):
        """Shift+left-click drag: horizontal/vertical pan."""
        yaw_r = math.radians(self.yaw)
        right = np.array([math.cos(yaw_r), 0, -math.sin(yaw_r)], dtype=np.float32)
        up = np.array([0, 1, 0], dtype=np.float32)
        speed = self.distance * 0.002
        self.target -= right * dx * speed
        self.target += up * dy * speed

    def scroll_zoom(self, delta: float):
        """Mouse wheel zoom."""
        factor = 1.0 - delta * 0.1
        self.distance = max(0.1, min(50.0, self.distance * factor))

    def get_view_matrix(self) -> np.ndarray:
        return _look_at(self.eye, self.target, np.array([0, 1, 0], dtype=np.float32))

    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect, self.near, self.far)

    def fit_to_bounds(self, bbox_min: np.ndarray, bbox_max: np.ndarray):
        """Auto-fit camera to model bounds."""
        center = (bbox_min + bbox_max) * 0.5
        size = np.linalg.norm(bbox_max - bbox_min)
        self.target = center.astype(np.float32)
        self.distance = max(size * 1.5, 0.5)
        self.yaw = 0.0
        self.pitch = 15.0


class RenderMesh:
    """GPU mesh data for rendering."""

    def __init__(self, vao, index_count: int, base_color: tuple, texture=None):
        self.vao = vao
        self.index_count = index_count
        self.base_color = base_color
        self.texture = texture


class ModelRenderer:
    """
    Offscreen 3D renderer for skinned meshes.

    Usage:
        renderer = ModelRenderer()
        renderer.load_model(fbx_data)
        image = renderer.render(800, 600, camera)
    """

    def __init__(self):
        if moderngl is None:
            raise ImportError("moderngl is required: pip install moderngl")

        self.ctx: moderngl.Context | None = None
        self.prog = None
        self.grid_prog = None
        self.fbo = None
        self._meshes: list[RenderMesh] = []
        self._grid_vao = None
        self._bone_count = 0
        self._bone_matrices = None  # (N, 4, 4) current bone transforms
        self._rest_matrices = None  # (N, 4, 4) rest pose inverse bind
        self._has_model = False
        self._fbo_size = (0, 0)

    def _ensure_context(self):
        if self.ctx is None:
            self.ctx = moderngl.create_standalone_context()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.CULL_FACE)
            self.prog = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=FRAGMENT_SHADER,
            )
            self.grid_prog = self.ctx.program(
                vertex_shader=GRID_VS,
                fragment_shader=GRID_FS,
            )
            self._build_grid()

    def _ensure_fbo(self, width: int, height: int):
        if self._fbo_size != (width, height):
            if self.fbo:
                self.fbo.release()
            color = self.ctx.texture((width, height), 4)
            depth = self.ctx.depth_renderbuffer((width, height))
            self.fbo = self.ctx.framebuffer(color_attachments=[color], depth_attachment=depth)
            self._fbo_size = (width, height)

    def _build_grid(self):
        """Build a ground-plane grid."""
        verts = []
        grid_size = 10
        grid_step = 0.5
        color_main = (0.35, 0.35, 0.35)
        color_axis_x = (0.6, 0.2, 0.2)
        color_axis_z = (0.2, 0.2, 0.6)

        for i in range(int(-grid_size / grid_step), int(grid_size / grid_step) + 1):
            x = i * grid_step
            c = color_axis_z if i == 0 else color_main
            verts.extend([x, 0, -grid_size, *c, x, 0, grid_size, *c])

        for i in range(int(-grid_size / grid_step), int(grid_size / grid_step) + 1):
            z = i * grid_step
            c = color_axis_x if i == 0 else color_main
            verts.extend([-grid_size, 0, z, *c, grid_size, 0, z, *c])

        data = np.array(verts, dtype='f4').tobytes()
        vbo = self.ctx.buffer(data)
        self._grid_vao = self.ctx.vertex_array(
            self.grid_prog,
            [(vbo, '3f 3f', 'in_position', 'in_color')],
        )
        self._grid_vert_count = len(verts) // 6

    def load_model(self, fbx_data):
        """Load model from FBXData for rendering."""
        from .fbx_loader import FBXData
        self._ensure_context()
        self._cleanup_model()

        bones = fbx_data.bones
        self._bone_count = len(bones)

        # Compute rest-pose global transforms and inverse bind matrices
        rest_global = np.zeros((self._bone_count, 4, 4), dtype=np.float32)
        for i, bone in enumerate(bones):
            if bone.parent_index < 0:
                rest_global[i] = bone.local_transform
            else:
                rest_global[i] = rest_global[bone.parent_index] @ bone.local_transform

        self._rest_global = rest_global.copy()
        self._rest_matrices = np.zeros_like(rest_global)
        for i, bone in enumerate(bones):
            ibm = bone.offset_matrix.copy()
            if np.allclose(ibm, np.eye(4)):
                ibm = np.linalg.inv(rest_global[i])
            self._rest_matrices[i] = ibm

        # Identity bone matrices (rest pose)
        self._bone_matrices = np.array(
            [rest_global[i] @ self._rest_matrices[i] for i in range(self._bone_count)],
            dtype=np.float32,
        )

        # Compute model bounding box
        all_positions = []

        for mesh_data in fbx_data.meshes:
            all_positions.append(mesh_data.positions)

            # Build VBO data: position(3) + normal(3) + texcoord(2) + joints(4) + weights(4)
            n = len(mesh_data.positions)
            vbo_data = np.zeros((n, 16), dtype=np.float32)
            vbo_data[:, 0:3] = mesh_data.positions
            vbo_data[:, 3:6] = mesh_data.normals
            vbo_data[:, 6:8] = mesh_data.texcoords
            vbo_data[:, 8:12] = mesh_data.joint_indices.astype(np.float32)
            vbo_data[:, 12:16] = mesh_data.joint_weights

            vbo = self.ctx.buffer(vbo_data.astype('f4').tobytes())
            ibo = self.ctx.buffer(mesh_data.indices.astype('i4').tobytes())

            vao = self.ctx.vertex_array(
                self.prog,
                [(vbo, '3f 3f 2f 4f 4f',
                  'in_position', 'in_normal', 'in_texcoord', 'in_joints', 'in_weights')],
                index_buffer=ibo,
                index_element_size=4,
            )

            # Material color
            mat_idx = min(mesh_data.material_index, len(fbx_data.materials) - 1)
            if mat_idx >= 0 and mat_idx < len(fbx_data.materials):
                color = fbx_data.materials[mat_idx].diffuse_color
            else:
                color = (0.8, 0.8, 0.8, 1.0)

            self._meshes.append(RenderMesh(
                vao=vao,
                index_count=len(mesh_data.indices),
                base_color=color,
            ))

        if all_positions:
            all_pos = np.concatenate(all_positions, axis=0)
            self._bbox_min = all_pos.min(axis=0)
            self._bbox_max = all_pos.max(axis=0)
        else:
            self._bbox_min = np.array([-1, -1, -1], dtype=np.float32)
            self._bbox_max = np.array([1, 1, 1], dtype=np.float32)

        self._has_model = True

    def get_bbox(self) -> tuple[np.ndarray, np.ndarray]:
        return self._bbox_min, self._bbox_max

    def set_bone_transforms(self, global_transforms: np.ndarray):
        """
        Update bone transforms for animation.
        global_transforms: (N, 4, 4) array of global bone transforms.
        """
        if self._rest_matrices is None:
            return
        n = min(len(global_transforms), self._bone_count)
        for i in range(n):
            self._bone_matrices[i] = global_transforms[i] @ self._rest_matrices[i]

    def reset_pose(self):
        """Reset to rest pose."""
        if self._rest_global is not None:
            for i in range(self._bone_count):
                self._bone_matrices[i] = self._rest_global[i] @ self._rest_matrices[i]

    @property
    def has_model(self) -> bool:
        return self._has_model

    @property
    def bone_count(self) -> int:
        return self._bone_count

    def render(self, width: int, height: int, camera: OrbitCamera) -> PILImage.Image:
        """Render the scene and return a PIL Image."""
        self._ensure_context()
        self._ensure_fbo(width, height)
        self.fbo.use()
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(0.18, 0.18, 0.20, 1.0)

        aspect = width / max(height, 1)
        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(aspect)
        vp = proj @ view

        # Draw grid
        self.grid_prog['u_vp'].write(vp.astype('f4').tobytes())
        self._grid_vao.render(moderngl.LINES)

        if self._has_model:
            model = np.eye(4, dtype=np.float32)
            self.prog['u_model'].write(model.tobytes())
            self.prog['u_view'].write(view.astype('f4').tobytes())
            self.prog['u_projection'].write(proj.astype('f4').tobytes())
            self.prog['u_light_dir'].value = (0.5, 1.0, 0.8)
            self.prog['u_light_color'].value = (0.9, 0.88, 0.85)
            self.prog['u_ambient'].value = (0.25, 0.25, 0.3)
            self.prog['u_camera_pos'].write(camera.eye.astype('f4').tobytes())
            self.prog['u_use_texture'].value = 0

            # Upload bone matrices
            if self._bone_count > 0 and self._bone_matrices is not None:
                self.prog['u_use_skinning'].value = 1
                for i in range(min(self._bone_count, 128)):
                    self.prog[f'u_bone_matrices[{i}]'].write(
                        self._bone_matrices[i].astype('f4').tobytes()
                    )
            else:
                self.prog['u_use_skinning'].value = 0

            for mesh in self._meshes:
                self.prog['u_base_color'].value = mesh.base_color[:4] if len(mesh.base_color) >= 4 else (*mesh.base_color[:3], 1.0)
                mesh.vao.render(moderngl.TRIANGLES)

        # Read pixels
        data = self.fbo.color_attachments[0].read()
        img = PILImage.frombytes('RGBA', (width, height), data)
        img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        return img

    def _cleanup_model(self):
        self._meshes.clear()
        self._has_model = False
        self._bone_count = 0
        self._bone_matrices = None
        self._rest_matrices = None
        self._rest_global = None

    def cleanup(self):
        self._cleanup_model()
        if self.ctx:
            self.ctx.release()
            self.ctx = None


# --- Math utilities ---

def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a view matrix (look-at)."""
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Compute a perspective projection matrix."""
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m
