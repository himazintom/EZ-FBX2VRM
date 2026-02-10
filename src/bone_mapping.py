"""
Mixamo bone name to VRM humanoid bone mapping.

VRM 0.x humanoid bone specification.
"""

# All VRM 0.x humanoid bones with required/optional status
VRM_REQUIRED_BONES = [
    "hips", "spine", "chest", "head", "neck",
    "leftUpperArm", "leftLowerArm", "leftHand",
    "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]

VRM_OPTIONAL_BONES = [
    "upperChest",
    "leftShoulder", "rightShoulder",
    "jaw", "leftEye", "rightEye",
    "leftToes", "rightToes",
    # Fingers
    "leftThumbProximal", "leftThumbIntermediate", "leftThumbDistal",
    "leftIndexProximal", "leftIndexIntermediate", "leftIndexDistal",
    "leftMiddleProximal", "leftMiddleIntermediate", "leftMiddleDistal",
    "leftRingProximal", "leftRingIntermediate", "leftRingDistal",
    "leftLittleProximal", "leftLittleIntermediate", "leftLittleDistal",
    "rightThumbProximal", "rightThumbIntermediate", "rightThumbDistal",
    "rightIndexProximal", "rightIndexIntermediate", "rightIndexDistal",
    "rightMiddleProximal", "rightMiddleIntermediate", "rightMiddleDistal",
    "rightRingProximal", "rightRingIntermediate", "rightRingDistal",
    "rightLittleProximal", "rightLittleIntermediate", "rightLittleDistal",
]

# Mixamo bone name -> VRM bone name
# Mixamo names may have prefixes like "mixamorig:", "mixamorig.", "mixamorig_", "mixamorig1:"
MIXAMO_TO_VRM = {
    "Hips": "hips",
    "Spine": "spine",
    "Spine1": "chest",
    "Spine2": "upperChest",
    "Neck": "neck",
    "Head": "head",
    "HeadTop_End": None,  # skip

    # Left arm
    "LeftShoulder": "leftShoulder",
    "LeftArm": "leftUpperArm",
    "LeftForeArm": "leftLowerArm",
    "LeftHand": "leftHand",

    # Right arm
    "RightShoulder": "rightShoulder",
    "RightArm": "rightUpperArm",
    "RightForeArm": "rightLowerArm",
    "RightHand": "rightHand",

    # Left leg
    "LeftUpLeg": "leftUpperLeg",
    "LeftLeg": "leftLowerLeg",
    "LeftFoot": "leftFoot",
    "LeftToeBase": "leftToes",
    "LeftToe_End": None,

    # Right leg
    "RightUpLeg": "rightUpperLeg",
    "RightLeg": "rightLowerLeg",
    "RightFoot": "rightFoot",
    "RightToeBase": "rightToes",
    "RightToe_End": None,

    # Left hand fingers
    "LeftHandThumb1": "leftThumbProximal",
    "LeftHandThumb2": "leftThumbIntermediate",
    "LeftHandThumb3": "leftThumbDistal",
    "LeftHandThumb4": None,
    "LeftHandIndex1": "leftIndexProximal",
    "LeftHandIndex2": "leftIndexIntermediate",
    "LeftHandIndex3": "leftIndexDistal",
    "LeftHandIndex4": None,
    "LeftHandMiddle1": "leftMiddleProximal",
    "LeftHandMiddle2": "leftMiddleIntermediate",
    "LeftHandMiddle3": "leftMiddleDistal",
    "LeftHandMiddle4": None,
    "LeftHandRing1": "leftRingProximal",
    "LeftHandRing2": "leftRingIntermediate",
    "LeftHandRing3": "leftRingDistal",
    "LeftHandRing4": None,
    "LeftHandPinky1": "leftLittleProximal",
    "LeftHandPinky2": "leftLittleIntermediate",
    "LeftHandPinky3": "leftLittleDistal",
    "LeftHandPinky4": None,

    # Right hand fingers
    "RightHandThumb1": "rightThumbProximal",
    "RightHandThumb2": "rightThumbIntermediate",
    "RightHandThumb3": "rightThumbDistal",
    "RightHandThumb4": None,
    "RightHandIndex1": "rightIndexProximal",
    "RightHandIndex2": "rightIndexIntermediate",
    "RightHandIndex3": "rightIndexDistal",
    "RightHandIndex4": None,
    "RightHandMiddle1": "rightMiddleProximal",
    "RightHandMiddle2": "rightMiddleIntermediate",
    "RightHandMiddle3": "rightMiddleDistal",
    "RightHandMiddle4": None,
    "RightHandRing1": "rightRingProximal",
    "RightHandRing2": "rightRingIntermediate",
    "RightHandRing3": "rightRingDistal",
    "RightHandRing4": None,
    "RightHandPinky1": "rightLittleProximal",
    "RightHandPinky2": "rightLittleIntermediate",
    "RightHandPinky3": "rightLittleDistal",
    "RightHandPinky4": None,
}

# Known Mixamo prefixes
MIXAMO_PREFIXES = [
    "mixamorig:",
    "mixamorig.",
    "mixamorig_",
    "mixamorig1:",
]


def strip_mixamo_prefix(bone_name: str) -> str:
    """Remove Mixamo prefix from bone name."""
    for prefix in MIXAMO_PREFIXES:
        if bone_name.startswith(prefix):
            return bone_name[len(prefix):]
    return bone_name


def get_vrm_bone_name(mixamo_bone_name: str) -> str | None:
    """
    Map a Mixamo bone name to VRM humanoid bone name.
    Returns None if the bone should be skipped or is not recognized.
    """
    stripped = strip_mixamo_prefix(mixamo_bone_name)
    return MIXAMO_TO_VRM.get(stripped)


def build_bone_mapping(bone_names: list[str]) -> dict[str, str]:
    """
    Build a mapping from source bone names to VRM bone names.
    Returns dict: {source_bone_name: vrm_bone_name}
    """
    mapping = {}
    for name in bone_names:
        vrm_name = get_vrm_bone_name(name)
        if vrm_name is not None:
            mapping[name] = vrm_name
    return mapping


def validate_bone_mapping(mapping: dict[str, str]) -> tuple[bool, list[str]]:
    """
    Validate that all required VRM bones are present in the mapping.
    Returns (is_valid, list_of_missing_required_bones).
    """
    mapped_vrm_bones = set(mapping.values())
    missing = [b for b in VRM_REQUIRED_BONES if b not in mapped_vrm_bones]
    return len(missing) == 0, missing
