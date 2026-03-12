"""Temporary utility: convert an uncompressed splat PLY to bundled SOG.

Run:
  python3 tests/convert_ply_to_sog.py --ply /path/to/splats_step030000.ply
  python3 tests/convert_ply_to_sog.py --ply /path/to/input.ply --out /path/to/output.sog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_output_path(ply_path: Path) -> Path:
    return ply_path.with_suffix(".sog")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an uncompressed FriendlySplat/gsplat PLY export to bundled SOG. CUDA is required for SOG export."
    )
    parser.add_argument(
        "--ply",
        required=True,
        help="Path to an uncompressed gsplat/FriendlySplat splat PLY.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .sog path. Defaults to <ply_basename>.sog next to the input file.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used while loading the PLY. Default: cpu",
    )
    parser.add_argument(
        "--cluster-device",
        default=None,
        help='Optional CUDA clustering device override for SH compression, e.g. "cuda:0". Default: auto',
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Fixed k-means iterations for SH clustering. Default: 10",
    )
    return parser


def _run() -> int:
    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))

    parser = _build_argparser()
    args = parser.parse_args()

    import torch

    from friendly_splat.modules.gaussian import GaussianModel
    from gsplat.exporter import export_splats

    ply_path = Path(args.ply).expanduser().resolve()
    if not ply_path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out is not None
        else _default_output_path(ply_path)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = GaussianModel.from_splat_ply(
        ply_path=str(ply_path),
        device=torch.device(str(args.device)),
        requires_grad=False,
    )

    export_splats(
        means=model.means.detach(),
        scales=model.log_scales.detach(),
        quats=model.quats.detach(),
        opacities=model.opacity_logits.detach(),
        sh0=model.sh0.detach(),
        shN=model.shN.detach(),
        format="sog",
        save_to=str(out_path),
        sog_iterations=args.iterations,
        sog_cluster_device=args.cluster_device,
    )

    print(f"Loaded PLY: {ply_path}")
    print(f"Saved bundled SOG: {out_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_run())
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from e
