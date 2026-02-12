"""Generate cluster VRM (flipped) and comparison (non-flipped) versions."""
from src.converter import convert_fbx_to_vrm

# Non-flipped version (with height scaling) for comparison
print("=== Generating non-flipped version ===")
convert_fbx_to_vrm(
    'T-Pose.fbx', 'T-Pose_noflip_170.vrm',
    target_height=1.7, flip_forward=False,
    callback=lambda m, p: print(f'[{p*100:5.1f}%] {m}'),
)
print('Done!')

# Flipped version for cluster
print("\n=== Generating cluster (flipped) version ===")
convert_fbx_to_vrm(
    'T-Pose.fbx', 'T-Pose_cluster.vrm',
    target_height=1.7, flip_forward=True,
    callback=lambda m, p: print(f'[{p*100:5.1f}%] {m}'),
)
print('Done!')
