"""Export the compiled swarm StateGraph as Mermaid source and a PNG image.

uv run --frozen python scripts/export_graph.py
uv run --frozen python scripts/export_graph.py --mermaid-only   # offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph import build_topology_graph

log = logging.getLogger("export_graph")

BASENAME = "swarm_architecture"


def export(out_dir: Path, *, png: bool) -> list[Path]:
    """Write `<out_dir>/swarm_architecture.mmd` (and `.png`); return the written paths."""
    drawable = build_topology_graph().get_graph()
    out_dir.mkdir(parents=True, exist_ok=True)

    mmd_path = out_dir / f"{BASENAME}.mmd"
    mmd_path.write_text(drawable.draw_mermaid(), "utf-8")
    written = [mmd_path]

    if png:
        png_path = out_dir / f"{BASENAME}.png"
        # Default draw method: the mermaid.ink HTTP API (network required).
        png_path.write_bytes(drawable.draw_mermaid_png(max_retries=3, retry_delay=2.0))
        written.append(png_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--mermaid-only", action="store_true", help="Skip the PNG render (no network needed)."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s | %(message)s")

    try:
        paths = export(args.output_dir, png=not args.mermaid_only)
    except (OSError, ValueError, ImportError) as exc:
        log.error("PNG render failed (%s); rerun with --mermaid-only to export offline.", exc)  # noqa: TRY400
        return 1
    for path in paths:
        log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
