"""
Code that uses the hierarchical localization toolbox (hloc)
to extract and match image features, estimate camera poses,
and do sparse reconstruction.
Requires hloc module from : https://github.com/cvg/Hierarchical-Localization
"""

# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional, Set, Tuple


class CameraModel(Enum):
    """Enum for camera types."""

    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"
    RADIAL = "RADIAL"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"
    OPENCV = "OPENCV"
    FULL_OPENCV = "FULL_OPENCV"


PANO_CONFIG = {
    "fov": 100.0,
    "views": [
        (0.0, 0.0),
        (90.0, 0.0),
        (180.0, 0.0),
        (-90.0, 0.0),
        (0.0, 90.0),
    ],
}


def _log(msg: str) -> None:
    print(str(msg), flush=True)


def get_rig_rotations() -> List["object"]:
    """Return panorama rig rotations defined in PANO_CONFIG."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotations: List[np.ndarray] = []
    for yaw, pitch in PANO_CONFIG["views"]:
        rot = Rotation.from_euler("XY", [-pitch, -yaw], degrees=True).as_matrix()
        rotations.append(rot)
    return rotations


def create_pano_rig_config(ref_idx: int = 0):
    """Create a pycolmap RigConfig describing the five-view panorama setup."""
    try:
        import pycolmap
    except ImportError:
        return None
    import numpy as np

    cams_from_pano = get_rig_rotations()
    rig_cameras = []
    for idx, cam_from_pano_R in enumerate(cams_from_pano):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_R = cam_from_pano_R @ cams_from_pano[ref_idx].T
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_R), np.zeros(3)
            )

        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=(idx == ref_idx),
                image_prefix=f"pano_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)


MODEL_FILENAMES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "frames.bin",
    "rigs.bin",
)


def _get_candidate_model_dirs(sparse_root: Path) -> List[Path]:
    """Return all folders under sparse_root that look like COLMAP models."""

    def looks_like_model_dir(path: Path) -> bool:
        return path.is_dir() and (path / "cameras.bin").exists()

    candidates: List[Path] = []
    seen: Set[Path] = set()

    def add_candidate(path: Path) -> None:
        if path in seen or not looks_like_model_dir(path):
            return
        seen.add(path)
        candidates.append(path)

    add_candidate(sparse_root)
    if sparse_root.exists():
        for cameras_file in sorted(sparse_root.rglob("cameras.bin")):
            add_candidate(cameras_file.parent)
    return candidates


def _copy_colmap_model(*, src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for filename in MODEL_FILENAMES:
        src_file = src / filename
        if not src_file.exists():
            continue
        dst_file = dst / filename
        if dst_file.exists():
            dst_file.unlink()
        shutil.copy2(src_file, dst_file)


def _select_largest_colmap_model(
    pycolmap_module: Any, sparse_root: Path
) -> Tuple[Path, int]:
    """Pick the model with the most registered images and copy it to sparse_root/0."""
    candidates = _get_candidate_model_dirs(sparse_root)
    if not candidates:
        raise RuntimeError(f"No COLMAP models found under {sparse_root}")

    model_stats: List[Tuple[Path, int]] = []
    for model_dir in candidates:
        try:
            reconstruction = pycolmap_module.Reconstruction(str(model_dir))
            model_stats.append((model_dir, reconstruction.num_reg_images()))
        except Exception as exc:
            _log(f"Warning: failed to read model {model_dir}: {exc}")
    if not model_stats:
        raise RuntimeError(
            f"Unable to read any COLMAP reconstructions under {sparse_root}"
        )

    best_dir, best_count = max(model_stats, key=lambda item: item[1])
    sparse0 = sparse_root / "0"
    if sparse0.exists():
        for filename in MODEL_FILENAMES:
            f = sparse0 / filename
            if f.exists():
                f.unlink()
    _copy_colmap_model(src=best_dir, dst=sparse0)
    return sparse0, int(best_count)


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Path exists (use --overwrite): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _ensure_empty_export_dirs(export_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    images_dir = export_dir / "images"
    sparse_dir = export_dir / "sparse" / "0"

    if images_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"images/ already exists under export dir (use --overwrite): {images_dir}"
            )
        shutil.rmtree(images_dir)
    if (export_dir / "sparse").exists():
        if not overwrite:
            raise FileExistsError(
                f"sparse/ already exists under export dir (use --overwrite): {export_dir / 'sparse'}"
            )
        shutil.rmtree(export_dir / "sparse")

    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, sparse_dir


def _prepare_export_root_for_undistort(export_dir: Path, overwrite: bool) -> None:
    images_dir = export_dir / "images"
    sparse_root = export_dir / "sparse"
    if images_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"images/ already exists under export dir (use --overwrite): {images_dir}"
            )
        shutil.rmtree(images_dir)
    if sparse_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"sparse/ already exists under export dir (use --overwrite): {sparse_root}"
            )
        shutil.rmtree(sparse_root)

    for script_name in ("run-colmap-geometric.sh", "run-colmap-photometric.sh"):
        script_path = export_dir / script_name
        if script_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Helper script already exists under export dir (use --overwrite): {script_path}"
                )
            try:
                script_path.unlink()
            except OSError:
                pass

    stereo_dir = export_dir / "stereo"
    if stereo_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"stereo/ already exists under export dir (use --overwrite): {stereo_dir}"
            )
        try:
            shutil.rmtree(stereo_dir)
        except OSError:
            pass


def run_hloc(
    *,
    image_dir: Path,
    hloc_dir: Path,
    sfm_root: Path,
    export_dir: Path,
    camera_model: CameraModel,
    verbose: bool = False,
    matching_method: Literal["exhaustive", "sequential", "retrieval"] = "sequential",
    feature_type: Literal[
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sift",
        "sosnet",
        "disk",
        "aliked-n16",
    ] = "superpoint_aachen",
    matcher_type: Literal[
        "superpoint+lightglue",
        "disk+lightglue",
        "aliked+lightglue",
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
    ] = "superglue",
    retrieval_type: Literal["netvlad", "megaloc", "dir", "openibl"] = "megaloc",
    num_matched: int = 50,
    refine_pixsfm: bool = False,
    use_single_camera_mode: bool = True,
    is_panorama: bool = False,
    overwrite: bool = False,
) -> None:
    """Run HLOC SfM and export a 3DGS-ready COLMAP scene under export_dir."""

    try:
        import pycolmap
    except ImportError:
        _log(
            "Error: missing dependency `pycolmap`. "
            "Install the sfm extra with "
            "`pip install -e \".[sfm]\" --no-build-isolation`."
        )
        sys.exit(1)

    try:
        from hloc import (  # type: ignore
            extract_features,
            match_features,
            pairs_from_exhaustive,
            pairs_from_retrieval,
            pairs_from_sequential,
            reconstruction,
        )
    except ImportError:
        _log(
            "Error: missing dependency `hloc`. "
            "Install HLOC exactly as described in tools/sfm/README.md "
            "(clone the repository with submodules, then run `pip install -e /path/to/Hierarchical-Localization`)."
        )
        sys.exit(1)

    try:
        from pixsfm.refine_hloc import PixSfM  # type: ignore
    except ImportError:
        PixSfM = None

    if refine_pixsfm and PixSfM is None:
        _log(
            "Error: refine_pixsfm=True requires `pixsfm`. "
            "See tools/sfm/README.md for the optional installation note."
        )
        sys.exit(1)

    if is_panorama and camera_model not in (
        CameraModel.PINHOLE,
        CameraModel.SIMPLE_PINHOLE,
    ):
        _log(
            "Warning: panorama mode currently supports PINHOLE/SIMPLE_PINHOLE only. "
            "Forcing camera_model=PINHOLE."
        )
        camera_model = CameraModel.PINHOLE

    _ensure_empty_dir(hloc_dir, overwrite=overwrite)
    _ensure_empty_dir(sfm_root, overwrite=overwrite)
    export_dir.mkdir(parents=True, exist_ok=True)

    sfm_pairs = hloc_dir / f"pairs-{retrieval_type}.txt"
    features = hloc_dir / "features.h5"
    matches = hloc_dir / "matches.h5"

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    references = sorted(
        [
            p.relative_to(image_dir).as_posix()
            for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]
    )
    if not references:
        raise RuntimeError(f"No images found under: {image_dir}")

    retrieval_conf = extract_features.confs[retrieval_type]  # type: ignore
    feature_conf = extract_features.confs[feature_type]  # type: ignore
    matcher_conf = match_features.confs[matcher_type]  # type: ignore

    extract_features.main(  # type: ignore
        feature_conf,
        image_dir,
        image_list=references,
        feature_path=features,
    )

    if matching_method == "exhaustive":
        pairs_from_exhaustive.main(  # type: ignore
            sfm_pairs,
            image_list=references,
            groupby_folder=is_panorama,
        )
    elif matching_method == "sequential":
        retrieval_path = extract_features.main(retrieval_conf, image_dir, hloc_dir)
        pairs_from_sequential.main(  # type: ignore
            output=sfm_pairs,
            image_list=references,
            window_size=8,
            quadratic_overlap=True,
            use_loop_closure=True,
            retrieval_path=retrieval_path,
            retrieval_interval=2,
            num_loc=3,
            groupby_folder=is_panorama,
        )
    elif matching_method == "retrieval":
        retrieval_path = extract_features.main(retrieval_conf, image_dir, hloc_dir)
        num_matched = min(len(references), int(num_matched))
        pairs_from_retrieval.main(  # type: ignore
            retrieval_path,
            sfm_pairs,
            num_matched=num_matched,
        )
    else:
        raise ValueError(f"Unknown matching_method: {matching_method!r}")

    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)  # type: ignore

    image_options = pycolmap.ImageReaderOptions(camera_model=camera_model.value)  # type: ignore
    mapper_opts: Optional[Any] = None
    rig_config = None

    try:
        mapper_opts_candidate = pycolmap.IncrementalPipelineOptions()
    except Exception:
        mapper_opts_candidate = None

    if mapper_opts_candidate is not None:
        mapper_opts = mapper_opts_candidate
    else:
        mapper_opts = {}

    def _assign_mapper_option(option_name: str, value: Any) -> None:
        nonlocal mapper_opts
        if mapper_opts is None:
            mapper_opts = {}
        if isinstance(mapper_opts, dict):
            mapper_opts[option_name] = value
        else:
            setattr(mapper_opts, option_name, value)

    if is_panorama:
        import numpy as np
        import cv2

        camera_mode = pycolmap.CameraMode.PER_FOLDER  # type: ignore
        rig_config = create_pano_rig_config()

        first_img_path = image_dir / references[0]
        img = cv2.imread(str(first_img_path))
        if img is None:
            raise RuntimeError(f"Cannot read calibration image: {first_img_path}")
        h, w = img.shape[:2]
        hfov_deg = float(PANO_CONFIG["fov"])
        hfov_rad = np.deg2rad(hfov_deg)
        focal = w / (2.0 * np.tan(hfov_rad / 2.0))
        cx, cy = w / 2.0 - 0.5, h / 2.0 - 0.5

        if camera_model == CameraModel.PINHOLE:
            image_options.camera_params = f"{focal},{focal},{cx},{cy}"
        elif camera_model == CameraModel.SIMPLE_PINHOLE:
            image_options.camera_params = f"{focal},{cx},{cy}"

        _assign_mapper_option("ba_refine_sensor_from_rig", False)
        _assign_mapper_option("ba_refine_focal_length", True)
        _assign_mapper_option("ba_refine_principal_point", False)
        _assign_mapper_option("ba_refine_extra_params", False)
    elif use_single_camera_mode:
        camera_mode = pycolmap.CameraMode.SINGLE  # type: ignore
    else:
        camera_mode = pycolmap.CameraMode.PER_IMAGE  # type: ignore

    if refine_pixsfm:
        assert PixSfM is not None
        sfm = PixSfM(  # type: ignore
            conf={
                "dense_features": {"use_cache": True},
                "KA": {
                    "dense_features": {"use_cache": True},
                    "max_kps_per_problem": 1000,
                },
                "BA": {"strategy": "costmaps"},
            }
        )
        refined, _ = sfm.reconstruction(
            sfm_root,
            image_dir,
            sfm_pairs,
            features,
            matches,
            image_list=references,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
        )
        _log(f"PixSfM refined: {refined.summary()}")
    else:
        reconstruction.main(  # type: ignore
            sfm_root,
            image_dir,
            sfm_pairs,
            features,
            matches,
            camera_mode=camera_mode,  # type: ignore
            image_options=image_options,
            verbose=verbose,
            mapper_options=mapper_opts or None,
            rig_config=rig_config,
        )

    sfm_dir, num_reg_images = _select_largest_colmap_model(pycolmap, sfm_root)
    _log(f"Selected model: {sfm_dir} ({num_reg_images} registered images)")

    # =========================================================================
    # Export: undistortion OR simple migration/flattening
    # =========================================================================
    if is_panorama:
        target_images_dir, target_sparse_dir = _ensure_empty_export_dirs(
            export_dir, overwrite=overwrite
        )
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        src_images = sorted(
            [
                p
                for p in image_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in valid_extensions
            ]
        )
        name_map: dict[str, str] = {}
        for src_path in src_images:
            rel_path = src_path.relative_to(image_dir)
            new_name = rel_path.as_posix().replace("/", "__")
            dst_path = target_images_dir / new_name
            shutil.copy2(src_path, dst_path)
            name_map[rel_path.as_posix()] = new_name

        recon = pycolmap.Reconstruction(str(sfm_dir))
        for image in recon.images.values():
            if image.name in name_map:
                image.name = name_map[image.name]
        recon.write(target_sparse_dir)
        _log(f"Exported panorama scene to: {export_dir}")
        return

    if camera_model in (CameraModel.PINHOLE, CameraModel.SIMPLE_PINHOLE):
        target_images_dir, target_sparse_dir = _ensure_empty_export_dirs(
            export_dir, overwrite=overwrite
        )
        shutil.copytree(image_dir, target_images_dir, dirs_exist_ok=True)
        _copy_colmap_model(src=sfm_dir, dst=target_sparse_dir)
        _log(f"Exported COLMAP scene to: {export_dir}")
        return

    _log(f"Running pycolmap.undistort_images for {camera_model.name}...")
    _prepare_export_root_for_undistort(export_dir, overwrite=overwrite)
    options = pycolmap.UndistortCameraOptions()  # type: ignore
    options.max_image_size = 2000
    pycolmap.undistort_images(  # type: ignore
        output_path=str(export_dir),
        input_path=str(sfm_dir),
        image_path=str(image_dir),
        output_type="COLMAP",
        undistort_options=options,
    )

    # Ensure undistorted sparse model ends up at sparse/0.
    sparse_root = export_dir / "sparse"
    if sparse_root.exists():
        sparse0 = sparse_root / "0"
        sparse0.mkdir(parents=True, exist_ok=True)
        for item in list(sparse_root.iterdir()):
            if item == sparse0:
                continue
            if item.suffix == ".bin":
                shutil.move(str(item), str(sparse0 / item.name))
    _log(f"Exported undistorted COLMAP scene to: {export_dir}")
