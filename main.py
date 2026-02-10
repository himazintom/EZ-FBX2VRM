"""
EZ-FBX2VRM - Convert Mixamo-rigged FBX files to VRM format.

Usage:
    GUI mode:   python main.py
    CLI mode:   python main.py --cli input.fbx [-o output.vrm]
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        description="EZ-FBX2VRM: Convert Mixamo-rigged FBX to VRM without Blender/Unity",
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Run in command-line mode (no GUI)",
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Input FBX file path (required in CLI mode)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output VRM file path (default: same name as input with .vrm extension)",
    )
    parser.add_argument(
        "--title", default="My Model",
        help="VRM model title",
    )
    parser.add_argument(
        "--author", default="Unknown",
        help="VRM model author",
    )
    parser.add_argument(
        "--license", default="CC0",
        help="VRM license (CC0, CC_BY, etc.)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.cli:
        # CLI mode
        if not args.input:
            parser.error("Input FBX file is required in CLI mode")

        from pathlib import Path
        from src.converter import convert_fbx_to_vrm

        output = args.output or str(Path(args.input).with_suffix('.vrm'))

        meta = {
            "title": args.title,
            "author": args.author,
            "license": args.license,
        }

        def progress(msg, pct):
            bar_len = 30
            filled = int(bar_len * pct)
            bar = '#' * filled + '-' * (bar_len - filled)
            print(f"\r[{bar}] {pct*100:5.1f}% {msg}", end="", flush=True)
            if pct >= 1.0:
                print()

        try:
            convert_fbx_to_vrm(args.input, output, meta=meta, callback=progress)
            print(f"\nOutput saved to: {output}")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # GUI mode
        from src.gui import main as gui_main
        gui_main()


if __name__ == "__main__":
    main()
