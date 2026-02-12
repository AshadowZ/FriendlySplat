import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    """
    Get a similarity transform to normalize a dataset from camera-to-world poses.

    This implementation follows a common normalization routine:
      1) Rotate the world so that z+ is the up axis (estimated from camera up vectors).
      2) Recenter the scene (either using "focus" or "poses" center).
      3) Rescale the scene using camera distances.

    Args:
        c2w: Camera-to-world matrices. Shape: (N, 4, 4).
        strict_scaling: If True, uses max distance for scaling; otherwise median.
        center_method: "focus" or "poses".

    Returns:
        transform: A 4x4 similarity transform.
        scale: Scalar scale factor applied inside `transform`.
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis.
    # Estimate the up axis by averaging the camera up axes.
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ as up axis, rotate 180deg about x.
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene.
    if center_method == "focus":
        # Find the closest point to the origin for each camera's center ray.
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        # Use the median of camera positions.
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances.
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale

    return transform, scale


def similarity_from_cameras_no_rotation(c2w, strict_scaling=False, center_method="focus"):
    """Get a similarity transform (translation + uniform scale only) from camera poses.

    This variant intentionally avoids any world rotation. It is useful when you want
    training space to stay aligned with the original COLMAP frame, so exported PLYs
    can be mapped back without rotating SH coefficients.
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    fwds = np.sum(R * np.array([0.0, 0.0, 1.0]), axis=-1)

    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(f"Unknown center_method {center_method}")

    transform = np.eye(4)
    transform[:3, 3] = translate

    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform, scale


def align_principal_axes(point_cloud):
    """Align principal axes of a point cloud using PCA.

    Args:
        point_cloud: Nx3 array of 3D points.

    Returns:
        transform: 4x4 SE(3) matrix that aligns the principal axes.
    """
    # Compute centroid.
    centroid = np.median(point_cloud, axis=0)

    # Translate point cloud to centroid.
    translated_point_cloud = point_cloud - centroid

    # Compute covariance matrix.
    covariance_matrix = np.cov(translated_point_cloud, rowvar=False)

    # Compute eigenvectors and eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues (descending order) so that the z-axis
    # is the principal axis with the smallest eigenvalue.
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]

    # Check orientation of eigenvectors. If determinant is negative, flip one axis.
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1

    # Create rotation matrix.
    rotation_matrix = eigenvectors.T

    # Create SE(3) matrix (4x4 transformation matrix).
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix, points):
    """Transform points using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix.
        points: Nx3 array of points.

    Returns:
        Nx3 array of transformed points.
    """
    assert matrix.shape == (4, 4)
    assert len(points.shape) == 2 and points.shape[1] == 3
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_cameras(matrix, camtoworlds):
    """Transform cameras using an SE(3) matrix.

    Args:
        matrix: 4x4 SE(3) matrix.
        camtoworlds: Nx4x4 array of camera-to-world matrices.

    Returns:
        Nx4x4 array of transformed camera-to-world matrices.
    """
    assert matrix.shape == (4, 4)
    assert len(camtoworlds.shape) == 3 and camtoworlds.shape[1:] == (4, 4)
    camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, matrix)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def transform_cameras_and_points(camtoworlds, points, *, rotate: bool = True):
    """Normalize cameras and points into a canonical frame.

    When `rotate=True` (default), this matches the original behavior:
      - Align world up axis (z-up)
      - Recentering
      - Uniform rescaling
      - Principal axes alignment (PCA)

    When `rotate=False`, this performs translation+uniform scale only (no rotation, no PCA).
    """
    if rotate:
        T1, scale = similarity_from_cameras(camtoworlds, strict_scaling=False)
        camtoworlds = transform_cameras(T1, camtoworlds)
        points = transform_points(T1, points)
        T2 = align_principal_axes(points)
        camtoworlds = transform_cameras(T2, camtoworlds)
        points = transform_points(T2, points)
        transform = T2 @ T1
        return camtoworlds, points, transform, scale

    T1, scale = similarity_from_cameras_no_rotation(camtoworlds, strict_scaling=False)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points = transform_points(T1, points)
    transform = T1
    return camtoworlds, points, transform, scale
