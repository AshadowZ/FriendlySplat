from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


_REQUIRED_SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def _logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(float(eps), 1.0 - float(eps))
    return torch.log(x / (1.0 - x))


def _rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    # Matches gsplat/exporter.py.
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def _knn_distances(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Return Euclidean KNN distances. Shape [N, k]."""
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "KNN-based scale initialization requires scikit-learn. "
            "Install it (e.g. `pip install scikit-learn` or `pip install -r friendly_splat/requirements.txt`)."
        ) from e

    x_np = points.detach().cpu().numpy()
    model = NearestNeighbors(n_neighbors=int(k), metric="euclidean").fit(x_np)
    distances, _indices = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(points)


def _load_splat_ply_uncompressed(*, ply_path: str) -> Dict[str, torch.Tensor]:
    """Load splat parameters from a gsplat/FriendlySplat uncompressed PLY export.

    Supported PLY:
      - `format binary_little_endian 1.0`
      - `element vertex N`
      - float properties include:
        - `x y z`
        - `f_dc_0 f_dc_1 f_dc_2`
        - `f_rest_*` (optional, flattened SH coefficients)
        - `opacity` (3DGS convention: opacity logits)
        - `scale_0 scale_1 scale_2` (3DGS convention: log-scales)
        - `rot_0 rot_1 rot_2 rot_3` (wxyz quaternion)

    Not supported:
      - `ply_compressed` exports (they have `element chunk`/`element sh` and quantized fields).
      - ASCII PLY.
    """
    ply_path = str(ply_path)
    path = Path(ply_path)
    if not path.is_file():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError("Unexpected EOF while reading PLY header.")
            s = line.decode("utf-8", errors="strict").rstrip("\n")
            header_lines.append(s)
            if s.strip() == "end_header":
                break

        if not header_lines or header_lines[0].strip() != "ply":
            raise ValueError("Not a PLY file (missing leading 'ply').")

        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_props: list[str] = []

        for s in header_lines:
            parts = s.strip().split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = " ".join(parts[1:])
                continue
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex:
                vertex_props.append(parts[-1])

        if fmt is None or fmt.strip() != "binary_little_endian 1.0":
            raise ValueError(
                f"Unsupported PLY format: {fmt!r}. Only 'binary_little_endian 1.0' is supported."
            )
        if any(s.strip().startswith("element chunk") for s in header_lines) or any(
            s.strip().startswith("element sh") for s in header_lines
        ):
            raise ValueError(
                "Unsupported PLY structure (looks like ply_compressed). "
                "Re-export with --io.ply-format ply (uncompressed)."
            )
        if vertex_count is None or int(vertex_count) <= 0:
            raise ValueError(f"Invalid vertex count: {vertex_count!r}.")
        if not vertex_props:
            raise ValueError("PLY has no vertex properties.")

        prop_to_idx = {name: i for i, name in enumerate(vertex_props)}

        def _req(name: str) -> int:
            if name not in prop_to_idx:
                raise KeyError(f"PLY is missing required vertex property {name!r}.")
            return int(prop_to_idx[name])

        ix = _req("x")
        iy = _req("y")
        iz = _req("z")
        idc0 = _req("f_dc_0")
        idc1 = _req("f_dc_1")
        idc2 = _req("f_dc_2")
        iop = _req("opacity")
        is0 = _req("scale_0")
        is1 = _req("scale_1")
        is2 = _req("scale_2")
        ir0 = _req("rot_0")
        ir1 = _req("rot_1")
        ir2 = _req("rot_2")
        ir3 = _req("rot_3")

        rest: list[tuple[int, int]] = []
        for name, idx in prop_to_idx.items():
            if name.startswith("f_rest_"):
                suffix = name[len("f_rest_") :]
                if suffix.isdigit():
                    rest.append((int(suffix), int(idx)))
        rest.sort(key=lambda t: t[0])
        rest_cols = [idx for _suffix, idx in rest]

        num_props = int(len(vertex_props))
        expected = int(vertex_count) * num_props
        data_bytes = f.read(int(expected) * 4)
        arr = np.frombuffer(
            data_bytes,
            dtype=np.dtype("<f4"),
            count=int(expected),
        )
        if int(arr.size) != int(expected):
            raise ValueError(
                f"PLY vertex data truncated: expected {expected} float32 values, got {arr.size}."
            )
        arr = arr.reshape(int(vertex_count), num_props)

        means = arr[:, [ix, iy, iz]]
        opacities = arr[:, iop]
        scales = arr[:, [is0, is1, is2]]
        quats = arr[:, [ir0, ir1, ir2, ir3]]
        sh0 = arr[:, [idc0, idc1, idc2]].reshape(int(vertex_count), 1, 3)

        if rest_cols:
            rest_flat = arr[:, rest_cols]
            if int(rest_flat.shape[1]) % 3 != 0:
                raise ValueError(
                    f"Invalid f_rest_* property count: {rest_flat.shape[1]} (must be divisible by 3)."
                )
            k = int(rest_flat.shape[1] // 3)
            shN = rest_flat.reshape(int(vertex_count), 3, k).transpose(0, 2, 1)
        else:
            shN = np.zeros((int(vertex_count), 0, 3), dtype=np.float32)

    return {
        "means": torch.from_numpy(means).float(),
        "scales": torch.from_numpy(scales).float(),
        "quats": torch.from_numpy(quats).float(),
        "opacities": torch.from_numpy(opacities).float(),
        "sh0": torch.from_numpy(sh0).float(),
        "shN": torch.from_numpy(shN).float(),
    }


def _build_gaussian_params(
    *,
    splats: Mapping[str, object],
    device: torch.device,
    requires_grad: bool,
    src: str,
) -> Dict[str, torch.nn.Parameter]:
    missing = [k for k in _REQUIRED_SPLAT_KEYS if k not in splats]
    if missing:
        raise ValueError(f"{src} splats missing required keys {missing}")

    tensors: Dict[str, torch.Tensor] = {}
    for key in _REQUIRED_SPLAT_KEYS:
        value = splats[key]
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        if not torch.is_tensor(value):
            raise ValueError(f"{src} splats['{key}'] is not a tensor: {type(value)!r}")
        tensors[key] = value.to(device=device, dtype=torch.float32).contiguous()

    n = int(tensors["means"].shape[0])
    expected_shapes = {
        "means": (n, 3),
        "scales": (n, 3),
        "quats": (n, 4),
        "opacities": (n,),
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != tuple(shape):
            raise ValueError(f"{src} invalid {key} shape: {tuple(tensors[key].shape)}")

    for key in ("sh0", "shN"):
        if tensors[key].dim() != 3 or int(tensors[key].shape[0]) != n:
            raise ValueError(f"{src} invalid {key} shape: {tuple(tensors[key].shape)}")

    return {
        k: torch.nn.Parameter(v.clone(), requires_grad=bool(requires_grad))
        for k, v in tensors.items()
    }


class GaussianModel(torch.nn.Module):
    """Trainable Gaussian parameters used by 3DGS training."""

    def __init__(self, params: Dict[str, torch.nn.Parameter]) -> None:
        super().__init__()
        self.params = torch.nn.ParameterDict(params)

    @property
    def splats(self) -> torch.nn.ParameterDict:
        return self.params

    @property
    def means(self) -> torch.nn.Parameter:
        return self.params["means"]

    @property
    def log_scales(self) -> torch.nn.Parameter:
        # Convention: store log-scales in `scales`.
        return self.params["scales"]

    @property
    def quats(self) -> torch.nn.Parameter:
        return self.params["quats"]

    @property
    def opacity_logits(self) -> torch.nn.Parameter:
        # Convention: store opacity logits in `opacities`.
        return self.params["opacities"]

    @property
    def sh0(self) -> torch.nn.Parameter:
        return self.params["sh0"]

    @property
    def shN(self) -> torch.nn.Parameter:
        return self.params["shN"]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        """3D Gaussian scales in linear space (exp of log-scales)."""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """3D Gaussian opacities in [0,1] (sigmoid of logits)."""
        return torch.sigmoid(self.opacity_logits)

    @property
    def num_sh_coeffs(self) -> int:
        """Total number of SH coefficients per Gaussian."""
        return 1 + int(self.shN.shape[1])

    @property
    def max_sh_degree(self) -> int:
        """Maximum SH degree supported by the stored SH coefficients."""
        total = int(self.num_sh_coeffs)
        root = math.isqrt(total)
        if root * root != total:
            raise ValueError(
                f"Invalid SH coefficient count: 1+shN={total} is not a perfect square."
            )
        max_degree = int(root) - 1
        if max_degree < 0:
            raise ValueError(
                f"Invalid SH coefficient count: 1+shN={total} yields max_degree={max_degree}."
            )
        return max_degree

    def sh_coeffs(self, *, sh_degree: int) -> torch.Tensor:
        """Return SH coefficients sliced to the active SH degree.

        Args:
            sh_degree: Active SH degree (0 uses only DC term).
        """
        sh_degree = int(sh_degree)
        if sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")
        if sh_degree > int(self.max_sh_degree):
            raise ValueError(
                f"sh_degree={sh_degree} exceeds max_sh_degree={self.max_sh_degree} "
                f"(num_sh_coeffs={self.num_sh_coeffs})."
            )
        if sh_degree == 0:
            return self.sh0
        sh_coeffs_full = torch.cat([self.sh0, self.shN], dim=1)
        active_k = (sh_degree + 1) ** 2
        return sh_coeffs_full[:, :active_k, :]

    def to_render_tensors(self, *, sh_degree: int) -> dict[str, torch.Tensor]:
        """Return tensors in the form expected by gsplat rasterization."""
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.sh_coeffs(sh_degree=int(sh_degree)),
        }

    def get_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Return named parameter groups for optimizer construction.

        This method intentionally returns only structural information (group name
        to parameters). Optimizer/scheduler policies live in trainer code.
        """
        return {name: [param] for name, param in self.splat_parameters().items()}

    def splat_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Return the canonical set of trainable splat parameters."""
        return {
            "means": self.means,
            "scales": self.log_scales,
            "quats": self.quats,
            "opacities": self.opacity_logits,
            "sh0": self.sh0,
            "shN": self.shN,
        }

    @classmethod
    def from_sfm(
        cls,
        *,
        points: torch.Tensor,
        points_rgb: torch.Tensor,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [N,3], got {tuple(points.shape)}")
        if points_rgb.shape != points.shape:
            raise ValueError(
                f"points_rgb must match points shape [N,3], got {tuple(points_rgb.shape)} vs {tuple(points.shape)}"
            )
        return cls._build_from_points(
            points=points.float(),
            rgbs=(points_rgb.float() / 255.0).clamp(0.0, 1.0),
            sh_degree=int(sh_degree),
            init_scale=float(init_scale),
            init_opacity=float(init_opacity),
            device=device,
        )

    @classmethod
    def from_random(
        cls,
        *,
        num_points: int,
        scene_scale: float,
        init_extent: float,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        n = int(num_points)
        if n <= 0:
            raise ValueError(f"num_points must be > 0, got {n}")
        extent = float(init_extent) * float(scene_scale)
        points = extent * (torch.rand((n, 3)) * 2.0 - 1.0)
        rgbs = torch.rand((n, 3))
        return cls._build_from_points(
            points=points,
            rgbs=rgbs,
            sh_degree=int(sh_degree),
            init_scale=float(init_scale),
            init_opacity=float(init_opacity),
            device=device,
        )

    @classmethod
    def from_ckpt(
        cls,
        *,
        ckpt_path: str,
        device: torch.device,
        requires_grad: bool = True,
    ) -> "GaussianModel":
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        if not isinstance(ckpt, dict):
            raise ValueError(
                f"Invalid checkpoint format (expected dict): {ckpt_path}"
            )
        splats = ckpt.get("splats", None)
        if not isinstance(splats, dict):
            raise ValueError(
                f"Checkpoint missing 'splats' dict for initialization: {ckpt_path}"
            )
        params = _build_gaussian_params(
            splats=splats,
            device=device,
            requires_grad=bool(requires_grad),
            src=f"ckpt={ckpt_path}",
        )
        return cls(params=params)

    @classmethod
    def from_splat_ply(
        cls,
        *,
        ply_path: str,
        device: torch.device,
        requires_grad: bool = True,
    ) -> "GaussianModel":
        """Initialize splats from a gsplat/FriendlySplat PLY export (`splats_step*.ply`)."""
        splats = _load_splat_ply_uncompressed(ply_path=str(ply_path))
        params = _build_gaussian_params(
            splats=splats,
            device=device,
            requires_grad=bool(requires_grad),
            src=f"ply={ply_path}",
        )
        return cls(params=params)

    @classmethod
    def _build_from_points(
        cls,
        *,
        points: torch.Tensor,
        rgbs: torch.Tensor,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        n = int(points.shape[0])
        if n <= 0:
            raise ValueError("Gaussian initialization requires at least one point.")

        if sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")

        # KNN includes self as nearest neighbor. Use K<=4 then drop [:, 0].
        if n < 2:
            dist_avg = torch.ones((n,), device=points.device, dtype=points.dtype)
        else:
            k = min(4, n)
            dists = _knn_distances(points, k=k)
            neighbor_dists = dists[:, 1:] if k > 1 else dists
            if int(neighbor_dists.numel()) == 0:
                dist_avg = torch.ones((n,), device=points.device, dtype=points.dtype)
            else:
                dist2_avg = (neighbor_dists**2).mean(dim=-1)
                dist_avg = torch.sqrt(dist2_avg).clamp(min=1e-8)
        scales = torch.log(dist_avg * float(init_scale)).unsqueeze(-1).repeat(1, 3)

        quats = torch.rand((n, 4))
        opacities = _logit(torch.full((n,), float(init_opacity), dtype=torch.float32))

        n_coeff = (sh_degree + 1) ** 2
        sh = torch.zeros((n, n_coeff, 3), dtype=torch.float32)
        sh[:, 0, :] = _rgb_to_sh(rgbs)

        params: Dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(points.to(device)),
            # Convention: store log-scales in `scales`.
            "scales": torch.nn.Parameter(scales.to(device)),
            "quats": torch.nn.Parameter(quats.to(device)),
            "opacities": torch.nn.Parameter(opacities.to(device)),
            "sh0": torch.nn.Parameter(sh[:, :1, :].to(device)),
            "shN": torch.nn.Parameter(sh[:, 1:, :].to(device)),
        }
        return cls(params=params)
