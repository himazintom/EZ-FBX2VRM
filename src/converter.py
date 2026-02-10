"""
High-level FBX to VRM conversion pipeline.
"""

import logging
from pathlib import Path

from .fbx_loader import load_fbx
from .vrm_builder import VRMBuilder

logger = logging.getLogger(__name__)


def convert_fbx_to_vrm(
    input_path: str,
    output_path: str,
    meta: dict | None = None,
    callback=None,
) -> None:
    """
    Convert an FBX file to VRM format.

    Args:
        input_path: Path to the input FBX file.
        output_path: Path to the output VRM file.
        meta: Optional VRM metadata (title, author, license, etc.)
        callback: Optional progress callback(message: str, progress: float).
    """
    def _progress(msg, pct):
        logger.info(msg)
        if callback:
            callback(msg, pct)

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_file.suffix.lower() == '.fbx':
        raise ValueError(f"Input file is not an FBX file: {input_path}")

    _progress("Starting conversion...", 0.0)

    # Phase 1: Load FBX
    _progress("Phase 1/3: Loading FBX...", 0.0)
    fbx_data = load_fbx(
        str(input_file),
        callback=lambda msg, p: _progress(f"  {msg}", p * 0.4),
    )

    _progress(
        f"  Loaded: {len(fbx_data.meshes)} mesh(es), "
        f"{len(fbx_data.bones)} bone(s), "
        f"{len(fbx_data.materials)} material(s)",
        0.4,
    )

    # Phase 2: Build VRM
    _progress("Phase 2/3: Building VRM...", 0.4)
    builder = VRMBuilder(fbx_data, fbx_path=str(input_file))
    vrm_data = builder.build(
        meta=meta,
        callback=lambda msg, p: _progress(f"  {msg}", 0.4 + p * 0.4),
    )

    # Phase 3: Write output
    _progress("Phase 3/3: Writing VRM file...", 0.8)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(vrm_data)

    size_mb = len(vrm_data) / (1024 * 1024)
    _progress(f"Done! Output: {output_file.name} ({size_mb:.1f} MB)", 1.0)
