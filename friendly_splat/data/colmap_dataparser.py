from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .colmap_io import get_extrinsic, get_intrinsics, read_model
from .base_dataparser import DataParser, DataparserOutputs
from .image_io import get_rel_paths, imread_rgb, resize_image_folder
from .scene_transform import transform_cameras_and_points


class ColmapDataParser(DataParser):
    """COLMAP dataparser.

    Produces image paths, camera-to-world matrices, intrinsics, and optional prior paths.
    """

    def __init__(
        self,
        *,
        data_dir: str,
        factor: int = 1,
        normalize_world_space: bool = False,
        test_every: int = 8,
        benchmark_train_split: bool = False,
        depth_dir_name: Optional[str] = None,
        normal_dir_name: Optional[str] = None,
        dynamic_mask_dir_name: Optional[str] = None,
        sky_mask_dir_name: Optional[str] = None,
    ):
        self.data_dir = str(data_dir)
        self.factor = int(factor)
        self.normalize_world_space = bool(normalize_world_space)
        self.test_every = int(test_every)
        self.benchmark_train_split = bool(benchmark_train_split)
        self.depth_dir_name = depth_dir_name
        self.normal_dir_name = normal_dir_name
        self.dynamic_mask_dir_name = dynamic_mask_dir_name
        self.sky_mask_dir_name = sky_mask_dir_name
        self._parse()

    def _parse(self) -> None:
        # 1) Load raw COLMAP reconstruction.
        cameras, images, points3d = self._load_colmap_model()
        if len(images) == 0:
            raise ValueError("No images found in COLMAP model.")

        # 2) Build per-image poses/intrinsics and 3D points.
        image_names, camtoworlds, camera_ids, Ks_dict, imsize_dict = self._extract_poses_and_intrinsics(
            cameras=cameras,
            images=images,
        )
        points, points_rgb = self._extract_points(points3d)
        # 3) Optionally normalize to a canonical world frame.
        camtoworlds, points, transform, scale = self._normalize_scene_if_needed(
            camtoworlds=camtoworlds,
            points=points,
        )
        # 4) Resolve image paths and validate intrinsics/image consistency.
        image_paths = self._resolve_image_paths(image_names=image_names)
        self._validate_intrinsics_image_resolution_matches(
            camera_ids=camera_ids,
            imsize_dict=imsize_dict,
            image_paths=image_paths,
        )
        Ks_per_image = np.stack([Ks_dict[camera_id].astype(np.float32) for camera_id in camera_ids], axis=0)

        self.transform = transform
        self.scale = scale
        self.scene_scale = self._compute_scene_scale(camtoworlds)
        self.points = points
        self.points_rgb = points_rgb
        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds.astype(np.float32)
        self.Ks = Ks_per_image

        if int(self.factor) > 1:
            configured_priors = [
                name
                for name, value in [
                    ("depth_dir_name", self.depth_dir_name),
                    ("normal_dir_name", self.normal_dir_name),
                    ("dynamic_mask_dir_name", self.dynamic_mask_dir_name),
                    ("sky_mask_dir_name", self.sky_mask_dir_name),
                ]
                if value is not None
            ]
            if configured_priors:
                raise ValueError(
                    f"factor={int(self.factor)} with auxiliary priors is not supported. "
                    f"Configured priors: {', '.join(configured_priors)}. "
                    "Please use factor=1 when enabling depth/normal/mask priors."
                )

        self.depth_paths = self._build_paths(self.depth_dir_name, ext=".npy")
        self.normal_paths = self._build_paths(self.normal_dir_name, ext=".png")
        self.dynamic_mask_paths = self._build_mask_paths(self.dynamic_mask_dir_name)
        self.sky_mask_paths = self._build_mask_paths(self.sky_mask_dir_name)

    def _resolve_colmap_dir(self, data_dir: str) -> Path:
        root = Path(data_dir)
        # Keep compatibility with common COLMAP layouts.
        candidates = [
            root / "sparse" / "0",
            root / "sparse",
            root / "colmap" / "sparse" / "0",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"COLMAP directory does not exist under {data_dir}.")

    def _load_colmap_model(self):
        colmap_dir = self._resolve_colmap_dir(self.data_dir)
        return read_model(str(colmap_dir))

    def _extract_poses_and_intrinsics(
        self,
        *,
        cameras,
        images,
    ) -> tuple[List[str], np.ndarray, List[int], Dict[int, np.ndarray], Dict[int, tuple[int, int]]]:
        factor = int(self.factor)

        c2w_mats = []
        camera_ids: List[int] = []
        Ks_dict: Dict[int, np.ndarray] = {}
        imsize_dict: Dict[int, tuple[int, int]] = {}
        warned_camera_ids: set[int] = set()

        for img in images.values():
            w2c = get_extrinsic(img)
            c2w = np.linalg.inv(w2c)
            c2w_mats.append(c2w)

            camera_id = int(img.camera_id)
            camera_ids.append(camera_id)

            cam = cameras[camera_id]
            K = get_intrinsics(cam)
            K[:2, :] /= float(factor)
            Ks_dict[camera_id] = K
            imsize_dict[camera_id] = (
                int(round(float(cam.width) / float(factor))),
                int(round(float(cam.height) / float(factor))),
            )

            type_ = cam.model
            if type_ not in ("SIMPLE_PINHOLE", "PINHOLE") and camera_id not in warned_camera_ids:
                warnings.warn(
                    f"COLMAP camera_id={camera_id} uses model={type_} (has distortion parameters). "
                    "This dataparser ignores distortion and treats the camera as pinhole using only "
                    "(fx, fy, cx, cy). Make sure your images are already undistorted before training.",
                    category=UserWarning,
                    stacklevel=2,
                )
                warned_camera_ids.add(camera_id)

        image_names = [img.name for img in images.values()]
        camtoworlds = np.stack(c2w_mats, axis=0)

        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]
        return image_names, camtoworlds, camera_ids, Ks_dict, imsize_dict

    def _extract_points(self, points3d) -> tuple[np.ndarray, np.ndarray]:
        points = np.array([pt.xyz for pt in points3d.values()], dtype=np.float32)
        points_rgb = np.array([pt.rgb for pt in points3d.values()], dtype=np.uint8)
        return points, points_rgb

    def _normalize_scene_if_needed(
        self,
        *,
        camtoworlds: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        if self.normalize_world_space:
            if points.shape[0] == 0:
                raise ValueError("normalize_world_space=True requires points3D in the COLMAP model.")
            camtoworlds, points, transform, scale = transform_cameras_and_points(camtoworlds, points)
            return camtoworlds, points, transform.astype(np.float32), float(scale)

        transform = np.eye(4, dtype=np.float32)
        scale = 1.0
        return camtoworlds, points, transform, scale

    def _compute_scene_scale(self, camtoworlds: np.ndarray) -> float:
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        return float(np.max(dists))

    def _resolve_image_dirs(self) -> tuple[Path, Path]:
        factor = int(self.factor)
        data_dir = Path(self.data_dir)
        colmap_image_dir = data_dir / "images"
        image_dir = data_dir / f"images_{factor}" if factor > 1 else colmap_image_dir

        if not colmap_image_dir.exists():
            raise FileNotFoundError(f"Image folder {colmap_image_dir} does not exist.")
        if factor <= 1 and not image_dir.exists():
            raise FileNotFoundError(f"Image folder {image_dir} does not exist.")
        return colmap_image_dir, image_dir

    def _resolve_image_paths(self, *, image_names: List[str]) -> List[str]:
        factor = int(self.factor)
        colmap_image_dir, image_dir = self._resolve_image_dirs()

        image_files = sorted(get_rel_paths(str(image_dir))) if image_dir.exists() else []
        # For downsampled data, reuse existing `images_{factor}` when available.
        # Otherwise, generate it from source images.
        if factor > 1 and len(image_files) == 0:
            resized_dir = resize_image_folder(
                str(colmap_image_dir),
                str(image_dir),
                factor=factor,
            )
            image_dir = Path(resized_dir)
            image_files = sorted(get_rel_paths(str(image_dir)))
        if len(image_files) == 0:
            raise FileNotFoundError(f"No images found under {image_dir}.")

        colmap_file_set = set(get_rel_paths(str(colmap_image_dir)))
        missing_in_colmap = [name for name in image_names if name not in colmap_file_set]
        if missing_in_colmap:
            sample = missing_in_colmap[0]
            raise FileNotFoundError(
                f"Image referenced by COLMAP model not found in source images dir: {sample}"
            )

        image_file_set = set(image_files)
        image_files_by_stem: Dict[str, List[str]] = {}
        for rel_path in image_files:
            stem = os.path.splitext(rel_path)[0]
            image_files_by_stem.setdefault(stem, []).append(rel_path)

        resolved_rel_paths: List[str] = []
        for name in image_names:
            # Fast path: exact relative-path match.
            if name in image_file_set:
                resolved_rel_paths.append(name)
                continue

            # Fallback path: same stem with extension change (e.g. jpg -> png).
            stem = os.path.splitext(name)[0]
            candidates = image_files_by_stem.get(stem, [])
            if len(candidates) == 1:
                resolved_rel_paths.append(candidates[0])
                continue
            if len(candidates) == 0:
                raise FileNotFoundError(
                    f"Could not resolve image path for COLMAP name '{name}' under {image_dir}"
                )
            raise ValueError(
                f"Ambiguous image resolution for COLMAP name '{name}' under {image_dir}: {candidates}"
            )

        return [str(image_dir / rel_path) for rel_path in resolved_rel_paths]

    def _validate_intrinsics_image_resolution_matches(
        self,
        *,
        camera_ids: List[int],
        imsize_dict: Dict[int, tuple[int, int]],
        image_paths: List[str],
    ) -> None:
        if len(camera_ids) != len(image_paths):
            raise ValueError(
                f"camera_ids/image_paths length mismatch: {len(camera_ids)} vs {len(image_paths)}."
            )
        if not image_paths:
            raise ValueError("No image paths found while validating intrinsics.")

        # Strict-mode check: validate only the first frame to avoid full-dataset image I/O.
        camera_id = int(camera_ids[0])
        image_path = image_paths[0]
        image = imread_rgb(image_path)
        actual_height, actual_width = image.shape[:2]
        expected_width, expected_height = imsize_dict[camera_id]
        if actual_width != expected_width or actual_height != expected_height:
            raise ValueError(
                "COLMAP intrinsics/image resolution mismatch detected (first image check). "
                f"idx=0, camera_id={camera_id}, image={image_path}, "
                f"expected=({expected_width}, {expected_height}), "
                f"actual=({actual_width}, {actual_height}), "
                f"factor={int(self.factor)}. "
                "Please regenerate COLMAP or use matching images/intrinsics."
            )

    def _build_paths(self, dir_name: Optional[str], ext: str) -> Optional[List[str]]:
        if dir_name is None:
            return None
        root = os.path.join(self.data_dir, dir_name)
        paths: List[str] = []
        for img_name in self.image_names:
            base_name, _ = os.path.splitext(img_name)
            paths.append(os.path.join(root, base_name + ext))

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            modality = "depth" if ext == ".npy" else "normal" if ext == ".png" else f"*{ext}"
            warnings.warn(
                f"{modality.capitalize()} directory {root} is missing {len(missing)}/{len(paths)} files. "
                f"Falling back to per-frame {modality} usage (missing frames will skip this prior).",
                category=UserWarning,
                stacklevel=2,
            )
        return paths

    def _build_mask_paths(self, dir_name: Optional[str]) -> Optional[List[str]]:
        if dir_name is None:
            return None
        mask_dir = os.path.join(self.data_dir, dir_name)
        paths: List[str] = []
        for img_name in self.image_names:
            base_name, _ext = os.path.splitext(img_name)
            candidate_same_ext = os.path.join(mask_dir, img_name)
            candidate_png = os.path.join(mask_dir, base_name + ".png")
            if os.path.exists(candidate_same_ext):
                paths.append(candidate_same_ext)
            elif os.path.exists(candidate_png):
                paths.append(candidate_png)
            else:
                paths.append(candidate_same_ext)

        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            warnings.warn(
                f"Mask directory {mask_dir} is missing {len(missing)}/{len(paths)} files. "
                "Falling back to per-frame mask usage (missing frames will skip this mask).",
                category=UserWarning,
                stacklevel=2,
            )
        return paths

    def get_dataparser_outputs(self, *, split: str) -> DataparserOutputs:
        split = str(split)
        N = int(len(self.image_names))
        indices = np.arange(N, dtype=np.int64)
        test_every = int(self.test_every)
        if test_every <= 0:
            raise ValueError(f"test_every must be > 0, got {test_every}.")

        if split == "train":
            if bool(self.benchmark_train_split):
                indices = indices[indices % test_every != 0]
        else:
            indices = indices[indices % test_every == 0]

        if int(indices.shape[0]) == 0:
            raise ValueError(
                f"Empty split '{split}' after applying test_every={test_every} "
                f"and benchmark_train_split={bool(self.benchmark_train_split)}."
            )

        return DataparserOutputs(
            image_names=list(self.image_names),
            image_paths=list(self.image_paths),
            camtoworlds=self.camtoworlds,
            Ks=self.Ks,
            split=split,
            indices=indices,
            scene_scale=float(self.scene_scale),
            transform=self.transform.astype(np.float32),
            scale=float(self.scale),
            points=self.points,
            points_rgb=self.points_rgb,
            depth_paths=self.depth_paths,
            normal_paths=self.normal_paths,
            dynamic_mask_paths=self.dynamic_mask_paths,
            sky_mask_paths=self.sky_mask_paths,
        )
