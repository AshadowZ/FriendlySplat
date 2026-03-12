"""Standalone smoke test for bundled SOG export.

Run:
  python3 tests/test_sog_export.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO


def _require_dependencies() -> tuple[object, object]:
    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"torch is required: {e}") from e

    try:
        import PIL  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"SOG export dependencies are required (Pillow): {e}") from e

    if not torch.cuda.is_available():
        raise RuntimeError("SOG export smoke test requires CUDA.")

    return torch, PIL


def _run() -> int:
    print("Running SOG export smoke test...")
    torch, _pil = _require_dependencies()

    from gsplat.exporter import export_splats

    n = 6
    means = torch.tensor(
        [
            [-1.0, 0.0, 0.5],
            [-0.5, 0.2, 0.1],
            [0.0, -0.3, 0.8],
            [0.4, 0.1, -0.2],
            [0.8, 0.6, 0.3],
            [1.2, -0.4, -0.7],
        ],
        dtype=torch.float32,
    )
    scales = torch.linspace(-1.0, 0.5, steps=n * 3, dtype=torch.float32).reshape(n, 3)
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9239, 0.3827, 0.0, 0.0],
            [0.9239, 0.0, 0.3827, 0.0],
            [0.9239, 0.0, 0.0, 0.3827],
            [0.7071, 0.7071, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    opacities = torch.tensor([-10.0, -1.2, -0.3, 0.2, 1.1, 2.0], dtype=torch.float32)
    sh0 = torch.linspace(-0.25, 0.25, steps=n * 3, dtype=torch.float32).reshape(n, 1, 3)
    shN = torch.linspace(-0.1, 0.1, steps=n * 15 * 3, dtype=torch.float32).reshape(
        n, 15, 3
    )

    sog_bytes = export_splats(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=shN,
        format="sog",
    )

    assert len(sog_bytes) > 0, "SOG export returned empty bytes"

    with zipfile.ZipFile(BytesIO(sog_bytes), mode="r") as zf:
        names = set(zf.namelist())
        required = {
            "meta.json",
            "means_l.webp",
            "means_u.webp",
            "quats.webp",
            "scales.webp",
            "sh0.webp",
            "shN_centroids.webp",
            "shN_labels.webp",
        }
        missing = required - names
        assert not missing, f"Missing files in bundled SOG: {sorted(missing)}"
        meta = json.loads(zf.read("meta.json").decode("utf-8"))

    assert meta["version"] == 2, f"Unexpected SOG version: {meta['version']}"
    assert meta["count"] == n, f"Unexpected splat count: {meta['count']}"
    assert meta["means"]["files"] == ["means_l.webp", "means_u.webp"]
    assert meta["scales"]["files"] == ["scales.webp"]
    assert meta["quats"]["files"] == ["quats.webp"]
    assert meta["sh0"]["files"] == ["sh0.webp"]
    assert meta["shN"]["files"] == ["shN_centroids.webp", "shN_labels.webp"]
    assert meta["shN"]["bands"] == 3, f"Unexpected SH band count: {meta['shN']['bands']}"
    assert meta["shN"]["count"] == 4, f"Unexpected SH palette count: {meta['shN']['count']}"

    print("PASS: bundled SOG export structure looks correct.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_run())
    except AssertionError as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        raise SystemExit(1) from e
