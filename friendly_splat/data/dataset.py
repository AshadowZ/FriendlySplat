from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .base_dataparser import DataparserOutputs
from .image_io import imread_gray, imread_rgb


def _infer_data_factor_from_image_path(path: str) -> int:
    # Best-effort: parse ".../images_2/..." -> 2. Default to 1.
    try:
        parts = str(path).replace("\\", "/").split("/")
        if len(parts) >= 2:
            parent = parts[-2]
            if parent.startswith("images_"):
                suffix = parent[len("images_") :]
                if suffix.isdigit():
                    v = int(suffix)
                    if v > 0:
                        return v
    except Exception:  # noqa: BLE001
        pass
    return 1


def _raise_prior_shape_mismatch(
    *,
    modality: str,
    path: str,
    got_shape: tuple[int, ...],
    expected_hw: tuple[int, int],
    image_path: str,
) -> None:
    h_exp, w_exp = int(expected_hw[0]), int(expected_hw[1])
    factor = _infer_data_factor_from_image_path(image_path)
    raise ValueError(
        f"{modality} prior shape mismatch for image={image_path!r} (factor={factor}). "
        f"Expected (H,W)=({h_exp},{w_exp}) but got {got_shape} from {path!r}. "
        f"Fix: generate priors/masks at images_{factor}/ resolution, or train with factor=1."
    )


class InputDataset(torch.utils.data.Dataset):
    """On-demand dataset backed by :class:`DataparserOutputs`.

    This class only consumes parsed scene outputs and provides the standard dataset interface.
    Parser creation/orchestration should happen outside this class.
    Path/file validation is intentionally skipped here; this dataset fully trusts
    parser-provided paths as already validated.
    """

    def __init__(self, parsed_scene: DataparserOutputs):
        super().__init__()
        self.parsed_scene = parsed_scene

        n = int(len(self.parsed_scene.image_names))
        if int(self.parsed_scene.camtoworlds.shape[0]) != n:
            raise ValueError(
                "DataparserOutputs.camtoworlds must have length N == len(image_names)."
            )
        if int(self.parsed_scene.Ks.shape[0]) != n:
            raise ValueError(
                "DataparserOutputs.Ks must have length N == len(image_names)."
            )
        if int(self.parsed_scene.indices.ndim) != 1:
            raise ValueError("DataparserOutputs.indices must be a 1D array.")

    @property
    def scene_scale(self) -> float:
        return float(self.parsed_scene.scene_scale)

    def __len__(self) -> int:
        return int(self.parsed_scene.indices.shape[0])

    def __getitem__(self, dataset_index: int) -> Dict[str, Any]:
        parsed_scene = self.parsed_scene
        # `dataset_index` is split-local index; convert to global image index first.
        image_index = int(parsed_scene.indices[int(dataset_index)])
        # Evaluation currently needs only RGB + camera tensors, so skip heavy prior I/O.
        load_auxiliary_priors = str(parsed_scene.split).lower() == "train"

        # Trust parser outputs directly: no per-sample path existence checks.
        image_path = parsed_scene.image_paths[image_index]
        image_arr = imread_rgb(image_path)
        h_img, w_img = int(image_arr.shape[0]), int(image_arr.shape[1])
        K = parsed_scene.Ks[image_index]
        camtoworld = parsed_scene.camtoworlds[image_index]

        depth_data = None
        if load_auxiliary_priors and parsed_scene.depth_paths is not None:
            depth_path = parsed_scene.depth_paths[image_index]
            # Depth prior is stored in the parser's normalized scene space; map back
            # with parser scale so training-side depth supervision matches render space.
            depth_data = np.load(depth_path).astype(np.float32)
            if depth_data.ndim == 3 and int(depth_data.shape[-1]) == 1:
                depth_data = depth_data[..., 0]
            if depth_data.ndim != 2 or (
                int(depth_data.shape[0]),
                int(depth_data.shape[1]),
            ) != (h_img, w_img):
                _raise_prior_shape_mismatch(
                    modality="depth",
                    path=str(depth_path),
                    got_shape=tuple(int(x) for x in depth_data.shape),
                    expected_hw=(h_img, w_img),
                    image_path=str(image_path),
                )
            depth_data = depth_data * float(parsed_scene.scale)

        normal_data = None
        if load_auxiliary_priors and parsed_scene.normal_paths is not None:
            normal_path = parsed_scene.normal_paths[image_index]
            normal_data = imread_rgb(normal_path)
            if (
                normal_data.ndim != 3
                or int(normal_data.shape[-1]) != 3
                or (int(normal_data.shape[0]), int(normal_data.shape[1]))
                != (h_img, w_img)
            ):
                _raise_prior_shape_mismatch(
                    modality="normal",
                    path=str(normal_path),
                    got_shape=tuple(int(x) for x in normal_data.shape),
                    expected_hw=(h_img, w_img),
                    image_path=str(image_path),
                )

        dynamic_mask_data = None
        if load_auxiliary_priors and parsed_scene.dynamic_mask_paths is not None:
            dyn_path = parsed_scene.dynamic_mask_paths[image_index]
            dyn = imread_gray(dyn_path)
            if dyn.ndim != 2 or (int(dyn.shape[0]), int(dyn.shape[1])) != (
                h_img,
                w_img,
            ):
                _raise_prior_shape_mismatch(
                    modality="dynamic_mask",
                    path=str(dyn_path),
                    got_shape=tuple(int(x) for x in dyn.shape),
                    expected_hw=(h_img, w_img),
                    image_path=str(image_path),
                )
            dynamic_mask_data = dyn > 0

        sky_mask_data = None
        if load_auxiliary_priors and parsed_scene.sky_mask_paths is not None:
            sky_path = parsed_scene.sky_mask_paths[image_index]
            sky = imread_gray(sky_path)
            if sky.ndim != 2 or (int(sky.shape[0]), int(sky.shape[1])) != (
                h_img,
                w_img,
            ):
                _raise_prior_shape_mismatch(
                    modality="sky_mask",
                    path=str(sky_path),
                    got_shape=tuple(int(x) for x in sky.shape),
                    expected_hw=(h_img, w_img),
                    image_path=str(image_path),
                )
            sky_mask_data = sky > 0

        image = torch.from_numpy(image_arr)
        image_id = torch.tensor(image_index, dtype=torch.long)
        depth_prior = (
            torch.from_numpy(depth_data).float().unsqueeze(-1)
            if depth_data is not None
            else torch.empty(0)
        )
        normal_prior_u8 = (
            torch.from_numpy(normal_data)
            if normal_data is not None
            else torch.empty(0, dtype=torch.uint8)
        )

        data: Dict[str, Any] = {
            # Key naming convention:
            # - explicit dtype suffix when ambiguous (`*_u8`, `*_f32`, `*_bool`)
            # - camera tensors keep canonical names (`K`, `camtoworld`, `image_id`)
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image_u8": image,
            "image_id": image_id,
            "depth_prior_f32": depth_prior,
            "normal_prior_u8": normal_prior_u8,
        }
        if dynamic_mask_data is not None:
            data["dynamic_mask_bool"] = torch.from_numpy(dynamic_mask_data).bool()
        if sky_mask_data is not None:
            data["sky_mask_bool"] = torch.from_numpy(sky_mask_data).bool()
        return data
