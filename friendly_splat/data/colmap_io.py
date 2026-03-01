import collections
import os
import struct

import numpy as np


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {
    camera_model.model_id: camera_model for camera_model in CAMERA_MODELS
}
CAMERA_MODEL_NAMES = {
    camera_model.model_name: camera_model for camera_model in CAMERA_MODELS
}


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        return True
    return False


_PLY_DTYPE_MAP = {
    "char": np.dtype("i1"),
    "int8": np.dtype("i1"),
    "uchar": np.dtype("u1"),
    "uint8": np.dtype("u1"),
    "short": np.dtype("<i2"),
    "int16": np.dtype("<i2"),
    "ushort": np.dtype("<u2"),
    "uint16": np.dtype("<u2"),
    "int": np.dtype("<i4"),
    "int32": np.dtype("<i4"),
    "uint": np.dtype("<u4"),
    "uint32": np.dtype("<u4"),
    "float": np.dtype("<f4"),
    "float32": np.dtype("<f4"),
    "double": np.dtype("<f8"),
    "float64": np.dtype("<f8"),
}


def read_points3d_ply(path_to_ply_file: str):
    """Read a point cloud from `points3D.ply` (binary_little_endian only).

    This is a lightweight fallback for datasets that ship `points3D.ply` instead of
    COLMAP's `points3D.txt/.bin`. Track information is not available and will be empty.
    """
    with open(path_to_ply_file, "rb") as f:
        fmt = None
        num_vertices = None
        in_vertex = False
        vertex_props = []
        while True:
            line = f.readline()
            if line == b"":
                raise ValueError("Unexpected EOF while reading PLY header.")
            s = line.decode("utf-8", errors="strict").strip()
            if s == "end_header":
                break
            parts = s.split()
            if not parts:
                continue
            if parts[0] == "format" and len(parts) >= 3:
                fmt = " ".join(parts[1:])
                continue
            if parts[0] == "element" and len(parts) >= 3:
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    num_vertices = int(parts[2])
                continue
            if parts[0] == "property" and in_vertex:
                if len(parts) >= 5 and parts[1] == "list":
                    raise ValueError(
                        "Unsupported PLY vertex property (list). Expected scalar properties only."
                    )
                if len(parts) < 3:
                    continue
                typ, name = parts[1], parts[2]
                if typ not in _PLY_DTYPE_MAP:
                    raise ValueError(f"Unsupported PLY property type: {typ}")
                vertex_props.append((name, _PLY_DTYPE_MAP[typ]))

        if fmt is None or fmt.strip() != "binary_little_endian 1.0":
            raise ValueError(
                f"Unsupported PLY format: {fmt!r}. Only 'binary_little_endian 1.0' is supported."
            )
        if num_vertices is None or int(num_vertices) <= 0:
            raise ValueError(f"Invalid PLY vertex count: {num_vertices!r}")
        if not vertex_props:
            raise ValueError("PLY has no vertex properties.")

        dtype = np.dtype(vertex_props)
        raw = f.read(int(num_vertices) * int(dtype.itemsize))
        expected_nbytes = int(num_vertices) * int(dtype.itemsize)
        if int(len(raw)) != expected_nbytes:
            raise ValueError(
                f"PLY vertex data truncated: expected {expected_nbytes} bytes, got {len(raw)}."
            )
        arr = np.frombuffer(raw, dtype=dtype, count=int(num_vertices))

    required = {"x", "y", "z", "red", "green", "blue"}
    missing = required - set(arr.dtype.names or ())
    if missing:
        raise KeyError(f"PLY is missing required properties: {sorted(missing)}")

    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
    empty_i = np.empty((0,), dtype=np.int64)
    return {
        int(i): Point3D(
            id=int(i),
            xyz=xyz[int(i)],
            rgb=rgb[int(i)],
            error=float(0.0),
            image_ids=empty_i,
            point2D_idxs=empty_i,
        )
        for i in range(int(num_vertices))
    }


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_cameras_text(path):
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2d,
                format_char_sequence="ddq" * num_points2d,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_images_text(path):
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3d_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3d_ids,
                )
    return images


def read_points3d_binary(path_to_model_file):
    points3d = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3d_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2d_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3d[point3d_id] = Point3D(
                id=point3d_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2d_idxs,
            )
    return points3d


def read_points3d_text(path):
    points3d = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3d_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                track_elems = elems[8:]
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2d_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3d[point3d_id] = Point3D(
                    id=point3d_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2d_idxs,
                )
    return points3d


def read_model(path, ext=""):
    path = str(path)
    if ext == "":
        if os.path.isfile(os.path.join(path, "cameras.bin")) and os.path.isfile(
            os.path.join(path, "images.bin")
        ):
            ext = ".bin"
        elif os.path.isfile(os.path.join(path, "cameras.txt")) and os.path.isfile(
            os.path.join(path, "images.txt")
        ):
            ext = ".txt"
        else:
            raise FileNotFoundError(
                "Cannot find cameras/images with .bin or .txt under: " + path
            )

    if ext == ".bin":
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3d_path = os.path.join(path, "points3D" + ext)
        if os.path.isfile(points3d_path):
            points3d = read_points3d_binary(points3d_path)
        elif os.path.isfile(os.path.join(path, "points3D.ply")):
            points3d = read_points3d_ply(os.path.join(path, "points3D.ply"))
        else:
            raise FileNotFoundError(
                "Cannot find points3D.bin/.txt/.ply under: " + path
            )
    elif ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3d_path = os.path.join(path, "points3D" + ext)
        if os.path.isfile(points3d_path):
            points3d = read_points3d_text(points3d_path)
        elif os.path.isfile(os.path.join(path, "points3D.ply")):
            points3d = read_points3d_ply(os.path.join(path, "points3D.ply"))
        else:
            raise FileNotFoundError(
                "Cannot find points3D.bin/.txt/.ply under: " + path
            )
    else:
        raise ValueError(f"Unknown model extension: {ext}")

    return cameras, images, points3d


def get_intrinsics(camera: Camera) -> np.ndarray:
    model = camera.model
    params = camera.params
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
        # fx = fy = params[0]
        K = np.array(
            [[params[0], 0.0, params[1]], [0.0, params[0], params[2]], [0.0, 0.0, 1.0]]
        )
        return K
    if model in (
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "RADIAL",
        "RADIAL_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    ):
        K = np.array(
            [[params[0], 0.0, params[2]], [0.0, params[1], params[3]], [0.0, 0.0, 1.0]]
        )
        return K
    raise ValueError(f"Unsupported camera model: {model}")


def get_extrinsic(image: Image) -> np.ndarray:
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)
    w2c = np.concatenate(
        [np.concatenate([R, t], axis=1), np.array([[0, 0, 0, 1]])], axis=0
    )
    return w2c
