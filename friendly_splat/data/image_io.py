from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np


def get_rel_paths(path_dir: str) -> List[str]:
    paths: List[str] = []
    for dp, _dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def imread_rgb(path: str) -> np.ndarray:
    """Read an image as RGB uint8 numpy array with shape [H, W, 3]."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("cv2.imread returned None")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        if int(img.shape[-1]) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif int(img.shape[-1]) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img[..., :3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imread_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("cv2.imread returned None")
    return img


def resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    os.makedirs(resized_dir, exist_ok=True)
    image_files = get_rel_paths(image_dir)
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(resized_dir, os.path.splitext(image_file)[0] + ".png")
        if os.path.isfile(resized_path):
            continue
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        image_arr = imread_rgb(image_path)
        resized_size = (
            int(round(image_arr.shape[1] / factor)),
            int(round(image_arr.shape[0] / factor)),
        )
        resized_image = cv2.resize(image_arr, resized_size, interpolation=cv2.INTER_CUBIC)
        resized_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(resized_path, resized_bgr):
            raise RuntimeError(f"cv2.imwrite failed to save resized image: {resized_path}")
    return resized_dir
