"""
Pose Solver: convert MediaPipe pose landmarks to VRM bone rotations.

MediaPipe Pose provides 33 body landmarks in 3D.
This module converts those landmark positions into bone rotation quaternions
that can drive a VRM humanoid skeleton.
"""

import math
import numpy as np

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_EYE_INNER = 1
LEFT_EYE = 2
LEFT_EYE_OUTER = 3
RIGHT_EYE_INNER = 4
RIGHT_EYE = 5
RIGHT_EYE_OUTER = 6
LEFT_EAR = 7
RIGHT_EAR = 8
MOUTH_LEFT = 9
MOUTH_RIGHT = 10
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0, 1, 0], dtype=np.float32)


def _quat_from_two_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Compute quaternion that rotates v_from to v_to.
    Returns [x, y, z, w].
    """
    a = _normalize(v_from)
    b = _normalize(v_to)
    dot = float(np.dot(a, b))

    if dot > 0.999999:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    if dot < -0.999999:
        # 180 degree rotation - find orthogonal axis
        perp = np.cross(a, np.array([1, 0, 0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0, 1, 0]))
        perp = _normalize(perp)
        return np.array([*perp, 0], dtype=np.float32)

    axis = np.cross(a, b)
    w = 1.0 + dot
    q = np.array([axis[0], axis[1], axis[2], w], dtype=np.float32)
    return q / np.linalg.norm(q)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x,y,z,w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=np.float32)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_to_mat4(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to 4x4 rotation matrix."""
    x, y, z, w = q
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1 - 2*(y*y + z*z)
    m[0, 1] = 2*(x*y - z*w)
    m[0, 2] = 2*(x*z + y*w)
    m[1, 0] = 2*(x*y + z*w)
    m[1, 1] = 1 - 2*(x*x + z*z)
    m[1, 2] = 2*(y*z - x*w)
    m[2, 0] = 2*(x*z - y*w)
    m[2, 1] = 2*(y*z + x*w)
    m[2, 2] = 1 - 2*(x*x + y*y)
    return m


def _slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    dot = float(np.dot(q1, q2))
    if dot < 0:
        q2 = -q2
        dot = -dot
    # Clamp to valid acos domain to prevent math domain errors from float imprecision
    dot = min(max(dot, 0.0), 1.0)
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        n = np.linalg.norm(result)
        return result / n if n > 1e-10 else q1.copy()
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-10:
        return q1.copy()
    a = math.sin((1 - t) * theta) / sin_theta
    b = math.sin(t * theta) / sin_theta
    result = a * q1 + b * q2
    n = np.linalg.norm(result)
    return result / n if n > 1e-10 else q1.copy()


# T-pose reference directions (Y-up, right-handed)
# These define the expected bone directions in the rest pose
REST_DIRECTIONS = {
    "spine":         np.array([0,  1,  0], dtype=np.float32),
    "chest":         np.array([0,  1,  0], dtype=np.float32),
    "upperChest":    np.array([0,  1,  0], dtype=np.float32),
    "neck":          np.array([0,  1,  0], dtype=np.float32),
    "head":          np.array([0,  1,  0], dtype=np.float32),
    "leftUpperArm":  np.array([ 1, 0,  0], dtype=np.float32),
    "leftLowerArm":  np.array([ 1, 0,  0], dtype=np.float32),
    "rightUpperArm": np.array([-1, 0,  0], dtype=np.float32),
    "rightLowerArm": np.array([-1, 0,  0], dtype=np.float32),
    "leftUpperLeg":  np.array([0, -1,  0], dtype=np.float32),
    "leftLowerLeg":  np.array([0, -1,  0], dtype=np.float32),
    "rightUpperLeg": np.array([0, -1,  0], dtype=np.float32),
    "rightLowerLeg": np.array([0, -1,  0], dtype=np.float32),
}


class PoseSolver:
    """
    Converts MediaPipe pose landmarks to VRM bone rotations.

    Call solve() each frame with the latest landmarks to get
    a dict of VRM bone name -> local rotation quaternion [x,y,z,w].
    """

    def __init__(self):
        self.smoothing = 0.4  # 0 = no smoothing, 1 = full smoothing
        self._prev_rotations: dict[str, np.ndarray] = {}

    def solve(self, landmarks: list) -> dict[str, np.ndarray]:
        """
        Solve bone rotations from MediaPipe pose landmarks.

        Args:
            landmarks: list of 33 landmarks, each with .x, .y, .z
                       (MediaPipe normalized coordinates, or world landmarks)

        Returns:
            dict of vrm_bone_name -> quaternion [x, y, z, w]
        """
        if not landmarks or len(landmarks) < 33:
            return {}

        # Convert landmarks to numpy array
        # MediaPipe uses: x=right, y=down, z=toward camera (screen coords)
        # We convert to: x=right, y=up, z=forward (standard 3D)
        pts = np.array(
            [[lm.x, -lm.y, -lm.z] for lm in landmarks],
            dtype=np.float32,
        )

        rotations: dict[str, np.ndarray] = {}

        # --- Hips position & rotation ---
        hip_center = (pts[LEFT_HIP] + pts[RIGHT_HIP]) * 0.5
        shoulder_center = (pts[LEFT_SHOULDER] + pts[RIGHT_SHOULDER]) * 0.5

        # Hips rotation: based on hip line direction
        hip_right = _normalize(pts[RIGHT_HIP] - pts[LEFT_HIP])
        spine_dir = _normalize(shoulder_center - hip_center)
        hip_forward = _normalize(np.cross(hip_right, spine_dir))
        hip_up = _normalize(np.cross(hip_forward, hip_right))
        # No explicit hips rotation for now (identity), apply via spine

        # --- Spine ---
        rotations["spine"] = _quat_from_two_vectors(
            REST_DIRECTIONS["spine"], spine_dir
        )

        # --- Chest ---
        chest_dir = _normalize(shoulder_center - hip_center)
        rotations["chest"] = _quat_from_two_vectors(
            REST_DIRECTIONS["chest"], chest_dir
        )

        # --- Neck ---
        neck_pos = shoulder_center
        head_pos = (pts[LEFT_EAR] + pts[RIGHT_EAR]) * 0.5
        neck_dir = _normalize(head_pos - neck_pos)
        rotations["neck"] = _quat_from_two_vectors(
            REST_DIRECTIONS["neck"], neck_dir
        )

        # --- Head ---
        nose_dir = _normalize(pts[NOSE] - head_pos)
        # Head tilt: use ear-to-ear line for roll, nose direction for pitch/yaw
        head_up = _normalize(pts[NOSE] - (pts[MOUTH_LEFT] + pts[MOUTH_RIGHT]) * 0.5)
        rotations["head"] = _quat_from_two_vectors(
            REST_DIRECTIONS["head"], _normalize(head_up + np.array([0, 0.5, 0]))
        )

        # --- Left Arm ---
        l_upper_dir = _normalize(pts[LEFT_ELBOW] - pts[LEFT_SHOULDER])
        rotations["leftUpperArm"] = _quat_from_two_vectors(
            REST_DIRECTIONS["leftUpperArm"], l_upper_dir
        )

        l_lower_dir = _normalize(pts[LEFT_WRIST] - pts[LEFT_ELBOW])
        # Lower arm rotation relative to upper arm
        parent_q = rotations["leftUpperArm"]
        parent_inv = _quat_inverse(parent_q)
        local_dir = _rotate_vec_by_quat(l_lower_dir, parent_inv)
        rotations["leftLowerArm"] = _quat_from_two_vectors(
            REST_DIRECTIONS["leftLowerArm"], local_dir
        )

        # --- Right Arm ---
        r_upper_dir = _normalize(pts[RIGHT_ELBOW] - pts[RIGHT_SHOULDER])
        rotations["rightUpperArm"] = _quat_from_two_vectors(
            REST_DIRECTIONS["rightUpperArm"], r_upper_dir
        )

        r_lower_dir = _normalize(pts[RIGHT_WRIST] - pts[RIGHT_ELBOW])
        parent_q = rotations["rightUpperArm"]
        parent_inv = _quat_inverse(parent_q)
        local_dir = _rotate_vec_by_quat(r_lower_dir, parent_inv)
        rotations["rightLowerArm"] = _quat_from_two_vectors(
            REST_DIRECTIONS["rightLowerArm"], local_dir
        )

        # --- Left Leg ---
        l_upper_leg_dir = _normalize(pts[LEFT_KNEE] - pts[LEFT_HIP])
        rotations["leftUpperLeg"] = _quat_from_two_vectors(
            REST_DIRECTIONS["leftUpperLeg"], l_upper_leg_dir
        )

        l_lower_leg_dir = _normalize(pts[LEFT_ANKLE] - pts[LEFT_KNEE])
        parent_q = rotations["leftUpperLeg"]
        parent_inv = _quat_inverse(parent_q)
        local_dir = _rotate_vec_by_quat(l_lower_leg_dir, parent_inv)
        rotations["leftLowerLeg"] = _quat_from_two_vectors(
            REST_DIRECTIONS["leftLowerLeg"], local_dir
        )

        # --- Right Leg ---
        r_upper_leg_dir = _normalize(pts[RIGHT_KNEE] - pts[RIGHT_HIP])
        rotations["rightUpperLeg"] = _quat_from_two_vectors(
            REST_DIRECTIONS["rightUpperLeg"], r_upper_leg_dir
        )

        r_lower_leg_dir = _normalize(pts[RIGHT_ANKLE] - pts[RIGHT_KNEE])
        parent_q = rotations["rightUpperLeg"]
        parent_inv = _quat_inverse(parent_q)
        local_dir = _rotate_vec_by_quat(r_lower_leg_dir, parent_inv)
        rotations["rightLowerLeg"] = _quat_from_two_vectors(
            REST_DIRECTIONS["rightLowerLeg"], local_dir
        )

        # --- Apply smoothing ---
        for bone_name, q in rotations.items():
            if bone_name in self._prev_rotations:
                q = _slerp(q, self._prev_rotations[bone_name], self.smoothing)
            rotations[bone_name] = q
            self._prev_rotations[bone_name] = q.copy()

        return rotations


def _rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a quaternion [x,y,z,w]."""
    qv = np.array([q[0], q[1], q[2]], dtype=np.float32)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (q[3] * uv + uuv)


def apply_rotations_to_skeleton(
    bone_rotations: dict[str, np.ndarray],
    bones: list,
    bone_name_to_index: dict[str, int],
    vrm_bone_mapping: dict[str, str],
    rest_global: np.ndarray,
) -> np.ndarray:
    """
    Apply solved rotations to the skeleton hierarchy.

    Args:
        bone_rotations: dict of vrm_bone_name -> quaternion [x,y,z,w]
        bones: list of BoneInfo from FBX
        bone_name_to_index: mapping of bone name -> index
        vrm_bone_mapping: mapping of source_bone_name -> vrm_bone_name
        rest_global: (N, 4, 4) rest-pose global transforms

    Returns:
        (N, 4, 4) new global transforms with applied rotations
    """
    n = len(bones)
    new_global = rest_global.copy()

    # Build reverse mapping: vrm_name -> bone_index
    vrm_to_idx: dict[str, int] = {}
    for src_name, vrm_name in vrm_bone_mapping.items():
        if src_name in bone_name_to_index:
            vrm_to_idx[vrm_name] = bone_name_to_index[src_name]

    # Apply rotations
    for vrm_name, quat in bone_rotations.items():
        if vrm_name not in vrm_to_idx:
            continue
        idx = vrm_to_idx[vrm_name]
        rot_mat = _quat_to_mat4(quat)

        # Apply rotation to rest-pose local transform
        bone = bones[idx]
        local = bone.local_transform.copy()
        # Extract translation, apply new rotation, keep scale
        t = local[:3, 3].copy()
        new_local = rot_mat.copy()
        new_local[:3, 3] = t

        if bone.parent_index >= 0:
            new_global[idx] = new_global[bone.parent_index] @ new_local
        else:
            new_global[idx] = new_local

    # Recompute children transforms
    for i in range(n):
        bone = bones[i]
        if bone.parent_index >= 0:
            vrm_name_for_bone = None
            for src, vn in vrm_bone_mapping.items():
                if bone_name_to_index.get(src) == i:
                    vrm_name_for_bone = vn
                    break

            if vrm_name_for_bone not in bone_rotations:
                # This bone wasn't directly rotated; recompute from parent
                new_global[i] = new_global[bone.parent_index] @ bone.local_transform

    return new_global
