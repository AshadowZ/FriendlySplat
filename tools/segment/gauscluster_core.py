from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import networkx as nx
import numpy as np
import open3d as o3d
import torch
from scipy.sparse import csr_matrix, vstack
from scipy.spatial import cKDTree
from scipy.stats import mode
from tqdm import tqdm

try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cupy as cp

    HAS_CUML = True
except ImportError:
    cuDBSCAN = None
    cp = None
    HAS_CUML = False


@dataclass
class Node:
    mask_list: list[tuple[int, int]]
    visible_frame: csr_matrix
    contained_mask: csr_matrix
    point_ids: np.ndarray
    node_info: tuple[int, int]
    son_node_info: set[tuple[int, int]] | None = None

    @staticmethod
    def create_node_from_list(
        node_list: list["Node"],
        node_info: tuple[int, int],
    ) -> "Node":
        mask_list: list[tuple[int, int]] = []
        point_ids_list: list[np.ndarray] = []
        son_node_info: set[tuple[int, int]] = set()
        visible_stack: list[csr_matrix] = []
        contained_stack: list[csr_matrix] = []

        for node in node_list:
            mask_list += node.mask_list
            point_ids_list.append(node.point_ids)
            son_node_info.add(node.node_info)
            visible_stack.append(node.visible_frame)
            contained_stack.append(node.contained_mask)

        if len(point_ids_list) == 1:
            point_ids = point_ids_list[0].copy()
        else:
            point_ids = point_ids_list[0]
            for arr in point_ids_list[1:]:
                point_ids = np.union1d(point_ids, arr)

        if len(visible_stack) == 1:
            visible_frame = visible_stack[0].copy()
        else:
            visible_frame = vstack(visible_stack).sum(axis=0)
            visible_frame[visible_frame > 1] = 1
            visible_frame = csr_matrix(visible_frame)

        if len(contained_stack) == 1:
            contained_mask = contained_stack[0].copy()
        else:
            contained_mask = vstack(contained_stack).sum(axis=0)
            contained_mask[contained_mask > 1] = 1
            contained_mask = csr_matrix(contained_mask)

        return Node(
            mask_list=mask_list,
            visible_frame=visible_frame,
            contained_mask=contained_mask,
            point_ids=point_ids,
            node_info=node_info,
            son_node_info=son_node_info,
        )


def read_mask(mask_path: str) -> np.ndarray:
    if mask_path.endswith(".npy"):
        mask = np.load(mask_path)
    else:
        mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def compute_mask_visible_frame(
    global_gaussian_in_mask_matrix: csr_matrix,
    gaussian_in_frame_matrix: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    print("[Mask Visibility] Computing mask visibility with sparse matrices ...")
    A = global_gaussian_in_mask_matrix.astype(np.float32)
    if not isinstance(A, csr_matrix):
        A = csr_matrix(A, dtype=np.float32)
    B = csr_matrix(gaussian_in_frame_matrix, dtype=np.float32)

    intersection_counts = A.T @ B
    mask_point_counts = np.array(A.sum(axis=0)).ravel() + 1e-6

    intersection_counts = intersection_counts.tocoo()
    visibility = (
        intersection_counts.data / mask_point_counts[intersection_counts.row]
    ) > float(threshold)

    result = csr_matrix(
        (
            np.ones(np.count_nonzero(visibility), dtype=bool),
            (
                intersection_counts.row[visibility],
                intersection_counts.col[visibility],
            ),
        ),
        shape=(A.shape[1], B.shape[1]),
    )
    print("[Mask Visibility] Completed.")
    return result.toarray()


def judge_single_mask(
    gaussian_in_mask_matrix: np.ndarray,
    mask_gaussian_pclds: Dict[str, np.ndarray],
    frame_mask_id: str,
    mask_visible_frame: np.ndarray,
    frame_mask_to_index: dict[tuple[int, int], int],
    num_global_masks: int,
    mask_visible_threshold: float = 0.7,
    contained_threshold: float = 0.8,
    undersegment_filter_threshold: float = 0.3,
) -> tuple[bool, np.ndarray, np.ndarray]:
    mask_gaussian_pcld = mask_gaussian_pclds[frame_mask_id]
    visible_frame = np.zeros(gaussian_in_mask_matrix.shape[1], dtype=bool)
    contained_mask = np.zeros(num_global_masks, dtype=bool)

    mask_gaussians_info = gaussian_in_mask_matrix[list(mask_gaussian_pcld), :]
    split_num = 0
    visible_num = 0

    for frame_id in np.where(mask_visible_frame)[0]:
        frame_info = mask_gaussians_info[:, frame_id]
        if frame_info.size == 0:
            continue
        mask_counts = np.bincount(frame_info)
        if mask_counts.size == 0:
            continue
        invalid_cnt = mask_counts[0] if mask_counts.shape[0] > 0 else 0
        total_cnt = frame_info.size
        if total_cnt == 0:
            continue
        if invalid_cnt / total_cnt > float(mask_visible_threshold):
            continue
        mask_counts[0] = 0
        if mask_counts.sum() == 0:
            continue
        best_mask_id = int(mask_counts.argmax())
        best_cnt = int(mask_counts[best_mask_id])
        valid_cnt = total_cnt - invalid_cnt
        if valid_cnt <= 0:
            continue
        visible_num += 1
        contained_ratio = best_cnt / valid_cnt
        if contained_ratio > float(contained_threshold):
            frame_mask_idx = frame_mask_to_index.get((int(frame_id), best_mask_id))
            if frame_mask_idx is None:
                continue
            contained_mask[frame_mask_idx] = True
            visible_frame[frame_id] = True
        else:
            split_num += 1

    is_valid = (
        visible_num > 0
        and split_num / max(visible_num, 1) <= float(undersegment_filter_threshold)
    )
    return is_valid, contained_mask, visible_frame


def get_observer_num_thresholds(visible_frames_sparse: csr_matrix) -> list[float]:
    observer_num_matrix = visible_frames_sparse @ visible_frames_sparse.T
    observer_num_list = observer_num_matrix.data
    observer_num_list = observer_num_list[observer_num_list > 0]
    if observer_num_list.size == 0:
        return [1.0]

    percentiles = np.arange(95, -5, -5)
    percentile_values = np.percentile(observer_num_list, percentiles)
    observer_num_thresholds: list[float] = []
    for percentile, observer_num in zip(percentiles, percentile_values):
        if observer_num <= 1:
            if percentile < 50:
                break
            observer_num = 1
        observer_num_thresholds.append(float(observer_num))
    return observer_num_thresholds


def update_graph(
    nodes: list[Node],
    observer_num_threshold: float,
    connect_threshold: float,
) -> nx.Graph:
    node_visible_frames = vstack([node.visible_frame for node in nodes])
    node_contained_masks = vstack([node.contained_mask for node in nodes])
    observer_nums = node_visible_frames @ node_visible_frames.T
    supporter_nums = node_contained_masks @ node_contained_masks.T
    observer_nums_dense = observer_nums.toarray()
    supporter_nums_dense = supporter_nums.toarray()
    view_consensus_rate = supporter_nums_dense / (observer_nums_dense + 1e-7)
    num_nodes = len(nodes)
    disconnect = np.eye(num_nodes, dtype=bool)
    disconnect = disconnect | (observer_nums_dense < float(observer_num_threshold))
    adjacency = view_consensus_rate >= float(connect_threshold)
    adjacency = adjacency & ~disconnect
    return nx.from_numpy_array(adjacency)


def cluster_into_new_nodes(
    iteration: int,
    old_nodes: list[Node],
    graph: nx.Graph,
) -> list[Node]:
    new_nodes: list[Node] = []
    for component in nx.connected_components(graph):
        node_info = (int(iteration), len(new_nodes))
        new_nodes.append(
            Node.create_node_from_list(
                [old_nodes[node] for node in component],
                node_info,
            )
        )
    return new_nodes


def iterative_clustering(
    nodes: list[Node],
    observer_num_thresholds: list[float],
    connect_threshold: float,
) -> list[Node]:
    iterator = tqdm(
        enumerate(observer_num_thresholds),
        total=len(observer_num_thresholds),
        desc="Iterative clustering",
    )
    for iterate_id, observer_num_threshold in iterator:
        graph = update_graph(nodes, observer_num_threshold, connect_threshold)
        nodes = cluster_into_new_nodes(iterate_id + 1, nodes, graph)
    return nodes


def iterative_cluster_masks(tracker: Dict) -> Dict:
    gaussian_in_frame_matrix = tracker["gaussian_in_frame_matrix"]
    mask_gaussian_pclds = tracker["mask_gaussian_pclds"]
    global_frame_mask_list = tracker["global_frame_mask_list"]

    frame_mask_to_index = {
        (int(frame_id), int(mask_id)): idx
        for idx, (frame_id, mask_id) in enumerate(global_frame_mask_list)
    }

    num_points = gaussian_in_frame_matrix.shape[0]
    gaussian_in_frame_maskid_matrix = np.zeros(
        (num_points, gaussian_in_frame_matrix.shape[1]),
        dtype=np.uint16,
    )

    mask_rows: list[np.ndarray] = []
    mask_cols: list[np.ndarray] = []

    for mask_idx, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        ids = mask_gaussian_pclds[f"{frame_id}_{mask_id}"]
        gaussian_in_frame_maskid_matrix[ids, frame_id] = mask_id
        if len(ids) == 0:
            continue
        ids_arr = np.asarray(ids, dtype=np.int64)
        mask_rows.append(ids_arr)
        mask_cols.append(np.full(ids_arr.shape[0], mask_idx, dtype=np.int64))

    if len(mask_rows) > 0:
        mask_row_idx = np.concatenate(mask_rows)
        mask_col_idx = np.concatenate(mask_cols)
    else:
        mask_row_idx = np.empty(0, dtype=np.int64)
        mask_col_idx = np.empty(0, dtype=np.int64)

    data = np.ones(mask_row_idx.shape[0], dtype=bool)
    global_gaussian_in_mask_matrix = csr_matrix(
        (data, (mask_row_idx, mask_col_idx)),
        shape=(num_points, len(global_frame_mask_list)),
    )

    mask_visible_frames = compute_mask_visible_frame(
        global_gaussian_in_mask_matrix,
        gaussian_in_frame_matrix,
    )

    contained_masks: list[np.ndarray] = []
    visible_frames: list[np.ndarray] = []
    undersegment_mask_ids: list[int] = []

    for mask_cnts, (frame_id, mask_id) in enumerate(
        tqdm(global_frame_mask_list, desc="Filter under-segmented masks")
    ):
        valid, contained_mask, visible_frame = judge_single_mask(
            gaussian_in_frame_maskid_matrix,
            mask_gaussian_pclds,
            f"{frame_id}_{mask_id}",
            mask_visible_frames[mask_cnts],
            frame_mask_to_index,
            len(global_frame_mask_list),
        )
        contained_masks.append(contained_mask)
        visible_frames.append(visible_frame)
        if not valid:
            undersegment_mask_ids.append(mask_cnts)

    contained_masks_arr = np.stack(contained_masks, axis=0)
    visible_frames_arr = np.stack(visible_frames, axis=0)
    undersegment_mask_id_set = set(undersegment_mask_ids)

    for global_mask_id in undersegment_mask_ids:
        frame_id, _ = global_frame_mask_list[global_mask_id]
        mask_projected_idx = np.where(contained_masks_arr[:, global_mask_id])[0]
        contained_masks_arr[:, global_mask_id] = False
        visible_frames_arr[mask_projected_idx, frame_id] = False

    contained_masks_sparse = csr_matrix(contained_masks_arr, dtype=np.int32)
    visible_frames_sparse = csr_matrix(visible_frames_arr, dtype=np.int32)
    contained_masks_sparse.sort_indices()
    visible_frames_sparse.sort_indices()

    print("[Perf] Starting co-occurrence matrix and observer threshold computation ...")
    threshold_t0 = time.perf_counter()
    observer_num_thresholds = get_observer_num_thresholds(visible_frames_sparse)
    print(
        f"[Perf] Co-occurrence/threshold computation took {time.perf_counter() - threshold_t0:.2f}s"
    )

    print("[Perf] Starting node list construction ...")
    node_t0 = time.perf_counter()
    nodes: list[Node] = []
    for global_mask_id, (frame_id, mask_id) in enumerate(global_frame_mask_list):
        if global_mask_id in undersegment_mask_id_set:
            continue
        mask_list = [(frame_id, mask_id)]
        frame = visible_frames_sparse.getrow(global_mask_id)
        frame_mask = contained_masks_sparse.getrow(global_mask_id)
        point_ids = np.unique(
            np.asarray(mask_gaussian_pclds[f"{frame_id}_{mask_id}"], dtype=np.int64)
        )
        node_info = (0, len(nodes))
        nodes.append(
            Node(
                mask_list=mask_list,
                visible_frame=frame,
                contained_mask=frame_mask,
                point_ids=point_ids,
                node_info=node_info,
            )
        )
    print(
        f"[Perf] Node construction took {time.perf_counter() - node_t0:.2f}s with {len(nodes)} nodes."
    )

    nodes = iterative_clustering(nodes, observer_num_thresholds, connect_threshold=0.9)

    tracker.update(
        {
            "nodes": nodes,
            "observer_num_thresholds": observer_num_thresholds,
            "undersegment_mask_ids": undersegment_mask_ids,
        }
    )
    return tracker


def _gpu_dbscan(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    points_cp = cp.asarray(points.astype(np.float32))
    model = cuDBSCAN(eps=float(eps), min_samples=int(min_points))
    labels = model.fit_predict(points_cp)
    if hasattr(labels, "values"):
        labels = labels.values
    if hasattr(labels, "__cuda_array_interface__"):
        labels = cp.asnumpy(labels)
    labels = np.asarray(labels, dtype=np.int32)
    return labels + 1


def dbscan_process(
    pcld: o3d.geometry.PointCloud,
    point_ids: list[int],
    eps: float = 0.1,
    min_points: int = 4,
    use_gpu: bool = False,
) -> tuple[list[o3d.geometry.PointCloud], list[np.ndarray]]:
    points_np = np.asarray(pcld.points)
    if use_gpu and not HAS_CUML:
        raise RuntimeError("cuML is not installed; GPU DBSCAN cannot be used.")
    if use_gpu and points_np.shape[0] >= int(min_points) and points_np.size > 0:
        labels = _gpu_dbscan(points_np, eps, min_points)
    else:
        labels = np.array(
            pcld.cluster_dbscan(eps=float(eps), min_points=int(min_points))
        ) + 1
    count = np.bincount(labels)
    pcld_list: list[o3d.geometry.PointCloud] = []
    point_ids_list: list[np.ndarray] = []
    pcld_ids_list = np.asarray(point_ids)
    for i in range(len(count)):
        remain_index = np.where(labels == i)[0]
        if len(remain_index) == 0:
            continue
        pcld_list.append(pcld.select_by_index(remain_index))
        point_ids_list.append(pcld_ids_list[remain_index])
    return pcld_list, point_ids_list


def merge_overlapping_objects(
    total_point_ids_list: list[np.ndarray],
    total_bbox_list: list[list[np.ndarray]],
    total_mask_list: list[list[tuple[int, int, float]]],
    overlapping_ratio: float = 0.8,
) -> tuple[list[np.ndarray], list[list[tuple[int, int, float]]]]:
    total_object_num = len(total_point_ids_list)
    invalid_object = np.zeros(total_object_num, dtype=bool)
    for i in range(total_object_num):
        if invalid_object[i]:
            continue
        point_ids_i = set(total_point_ids_list[i])
        bbox_i = total_bbox_list[i]
        for j in range(i + 1, total_object_num):
            if invalid_object[j]:
                continue
            point_ids_j = set(total_point_ids_list[j])
            bbox_j = total_bbox_list[j]
            overlap_bbox = True
            for k in range(3):
                if bbox_i[0][k] > bbox_j[1][k] or bbox_j[0][k] > bbox_i[1][k]:
                    overlap_bbox = False
                    break
            if not overlap_bbox:
                continue
            intersect = len(point_ids_i.intersection(point_ids_j))
            if intersect / len(point_ids_i) > float(overlapping_ratio):
                invalid_object[i] = True
            elif intersect / len(point_ids_j) > float(overlapping_ratio):
                invalid_object[j] = True

    valid_point_ids_list: list[np.ndarray] = []
    valid_mask_list: list[list[tuple[int, int, float]]] = []
    for i in range(total_object_num):
        if not invalid_object[i]:
            valid_point_ids_list.append(total_point_ids_list[i])
            valid_mask_list.append(total_mask_list[i])
    return valid_point_ids_list, valid_mask_list


def filter_point(
    point_frame_matrix: np.ndarray,
    node: Node,
    pcld_list: list[o3d.geometry.PointCloud],
    point_ids_list: list[np.ndarray],
    mask_point_clouds: Dict[str, np.ndarray],
    point_filter_threshold: float,
) -> tuple[list[np.ndarray], list[list[np.ndarray]], list[list[tuple[int, int, float]]]]:
    node_global_frame_id_list = np.where(node.visible_frame.toarray().ravel() > 0)[0]
    mask_list = node.mask_list

    point_appear_in_video_nums: list[np.ndarray] = []
    point_appear_in_node_matrices: list[np.ndarray] = []
    for point_ids in point_ids_list:
        point_appear_in_video_matrix = point_frame_matrix[point_ids, :]
        point_appear_in_video_matrix = point_appear_in_video_matrix[
            :,
            node_global_frame_id_list,
        ]
        point_appear_in_video_nums.append(np.sum(point_appear_in_video_matrix, axis=1))
        point_appear_in_node_matrices.append(
            np.zeros_like(point_appear_in_video_matrix, dtype=bool)
        )

    object_mask_list: list[list[tuple[int, int, float]]] = [
        [] for _ in range(len(point_ids_list))
    ]
    for frame_id, mask_id in mask_list:
        if frame_id not in node_global_frame_id_list:
            continue
        frame_id_in_list = np.where(node_global_frame_id_list == frame_id)[0][0]
        mask_point_ids = list(mask_point_clouds[f"{frame_id}_{mask_id}"])
        for idx_obj, point_ids in enumerate(point_ids_list):
            point_ids_within_object = np.where(np.isin(point_ids, mask_point_ids))[0]
            point_appear_in_node_matrices[idx_obj][
                point_ids_within_object,
                frame_id_in_list,
            ] = True
            if len(point_ids_within_object) > 0:
                object_mask_list[idx_obj].append(
                    (
                        frame_id,
                        mask_id,
                        len(point_ids_within_object) / len(point_ids),
                    )
                )

    filtered_point_ids: list[np.ndarray] = []
    filtered_bbox_list: list[list[np.ndarray]] = []
    filtered_mask_list: list[list[tuple[int, int, float]]] = []
    for i, (point_appear_in_video_num, point_appear_in_node_matrix) in enumerate(
        zip(point_appear_in_video_nums, point_appear_in_node_matrices)
    ):
        detection_ratio = np.sum(point_appear_in_node_matrix, axis=1) / (
            point_appear_in_video_num + 1e-6
        )
        valid_point_ids = np.where(detection_ratio > float(point_filter_threshold))[0]
        if len(valid_point_ids) == 0 or len(object_mask_list[i]) < 2:
            continue
        filtered_point_ids.append(point_ids_list[i][valid_point_ids])
        points_np = np.asarray(pcld_list[i].points)[valid_point_ids]
        filtered_bbox_list.append(
            [
                np.amin(points_np, axis=0),
                np.amax(points_np, axis=0),
            ]
        )
        filtered_mask_list.append(object_mask_list[i])
    return filtered_point_ids, filtered_bbox_list, filtered_mask_list


def post_process_clusters(
    tracker: Dict,
    point_positions: torch.Tensor,
    point_filter_threshold: float = 0.5,
    dbscan_eps: float = 0.1,
    dbscan_min_points: int = 4,
    overlap_ratio: float = 0.8,
    use_gpu_dbscan: bool = False,
) -> Dict:
    nodes = tracker["nodes"]
    mask_gaussian_pclds = tracker["mask_gaussian_pclds"]
    gaussian_in_frame_matrix = tracker["gaussian_in_frame_matrix"]

    total_point_ids_list: list[np.ndarray] = []
    total_bbox_list: list[list[np.ndarray]] = []
    total_mask_list: list[list[tuple[int, int, float]]] = []
    scene_points = point_positions.detach().cpu().numpy()

    iterator = tqdm(nodes, total=len(nodes), desc="DBScan+point filtering")
    for node in iterator:
        if len(node.mask_list) < 2:
            continue
        node_point_ids = node.point_ids.tolist()
        pcld = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(scene_points[node_point_ids])
        )
        pcld_list, point_ids_list = dbscan_process(
            pcld,
            node_point_ids,
            eps=dbscan_eps,
            min_points=dbscan_min_points,
            use_gpu=use_gpu_dbscan,
        )
        point_ids_list, bbox_list, mask_list = filter_point(
            gaussian_in_frame_matrix,
            node,
            pcld_list,
            point_ids_list,
            mask_gaussian_pclds,
            point_filter_threshold,
        )
        total_point_ids_list.extend(point_ids_list)
        total_bbox_list.extend(bbox_list)
        total_mask_list.extend(mask_list)

    total_point_ids_list, total_mask_list = merge_overlapping_objects(
        total_point_ids_list,
        total_bbox_list,
        total_mask_list,
        overlapping_ratio=overlap_ratio,
    )

    tracker.update(
        {
            "total_point_ids_list": total_point_ids_list,
            "total_mask_list": total_mask_list,
        }
    )
    return tracker


def remedy_undersegment(
    tracker: Dict,
    threshold: float = 0.8,
) -> Dict:
    undersegment_frame_masks = [
        tracker["global_frame_mask_list"][fid]
        for fid in tracker["undersegment_mask_ids"]
    ]
    error_undersegment_frame_masks: dict[tuple[int, int], int] = {}
    remedy_undersegment_frame_masks: list[int] = []

    total_instances = len(tracker["total_point_ids_list"])
    if total_instances == 0:
        tracker["undersegment_mask_ids"] = remedy_undersegment_frame_masks
        return tracker

    num_points = tracker["gaussian_in_frame_matrix"].shape[0]
    gaussian_to_instance = np.full(num_points, -1, dtype=np.int32)
    for inst_idx, point_ids in enumerate(tracker["total_point_ids_list"]):
        if len(point_ids) == 0:
            continue
        gaussian_to_instance[np.asarray(point_ids, dtype=np.int64)] = inst_idx

    frame_mask_to_index = {
        tuple(frame_mask): idx
        for idx, frame_mask in enumerate(tracker["global_frame_mask_list"])
    }

    for frame_mask in tqdm(undersegment_frame_masks, desc="Fix under-segmented masks"):
        frame_id, mask_id = frame_mask
        frame_mask_gaussian = tracker["mask_gaussian_pclds"][f"{frame_id}_{mask_id}"]
        if len(frame_mask_gaussian) == 0:
            remedy_undersegment_frame_masks.append(
                frame_mask_to_index[tuple(frame_mask)]
            )
            continue
        instance_ids = gaussian_to_instance[
            np.asarray(frame_mask_gaussian, dtype=np.int64)
        ]
        instance_ids = instance_ids[instance_ids >= 0]
        if instance_ids.size == 0:
            remedy_undersegment_frame_masks.append(
                frame_mask_to_index[tuple(frame_mask)]
            )
            continue
        counts = np.bincount(instance_ids, minlength=total_instances)
        best_match_instance_idx = int(counts.argmax())
        best_match_intersect = int(counts[best_match_instance_idx])
        if best_match_intersect / len(frame_mask_gaussian) > float(threshold):
            error_undersegment_frame_masks[frame_mask] = best_match_instance_idx
        else:
            remedy_undersegment_frame_masks.append(
                frame_mask_to_index[tuple(frame_mask)]
            )

    tracker["undersegment_mask_ids"] = remedy_undersegment_frame_masks
    total_mask_list = tracker["total_mask_list"]
    for frame_mask, instance_idx in error_undersegment_frame_masks.items():
        total_mask_list[instance_idx].append(frame_mask)
    tracker["total_mask_list"] = total_mask_list
    return tracker


def export_color_cluster(
    tracker: Dict,
    point_positions: torch.Tensor,
    save_dir: Path,
    filename: str = "color_cluster.ply",
    assign_unlabeled_knn: bool = True,
    knn_k: int = 1,
    knn_filename: str = "color_cluster_knn.ply",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    total_point_ids_list = tracker.get("total_point_ids_list", [])
    if len(total_point_ids_list) == 0:
        print("[Export] total_point_ids_list is empty, skip export.")
        return

    xyz = point_positions.detach().cpu().numpy()
    num_points = xyz.shape[0]
    colors = np.zeros((num_points, 3), dtype=np.float32)
    inst_labels = np.full(num_points, -1, dtype=np.int32)
    rng = np.random.default_rng(0)
    inst_colors = rng.random((len(total_point_ids_list), 3)) * 0.7 + 0.3

    for idx, point_ids in enumerate(total_point_ids_list):
        pts = np.asarray(point_ids, dtype=int)
        colors[pts] = inst_colors[idx]
        inst_labels[pts] = idx

    pcld = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcld.colors = o3d.utility.Vector3dVector(colors)
    save_path = save_dir / filename
    o3d.io.write_point_cloud(str(save_path), pcld)
    print(f"[Export] Saved colored instance point cloud to {save_path}")
    label_path = save_dir / "instance_labels.npy"
    np.save(label_path, inst_labels.copy())
    print(f"[Export] Saved instance labels to {label_path}")

    if not assign_unlabeled_knn:
        return

    unlabeled_mask = inst_labels < 0
    assigned_mask = ~unlabeled_mask
    if not np.any(unlabeled_mask):
        print(
            "[Export] All points already have instance colors; skipping KNN recoloring."
        )
        return
    if not np.any(assigned_mask):
        print("[Export] No reference instance points available for KNN; skipping.")
        return

    k = max(1, min(int(knn_k), int(np.sum(assigned_mask))))
    assigned_xyz = xyz[assigned_mask]
    assigned_labels = inst_labels[assigned_mask]
    tree = cKDTree(assigned_xyz)
    _, nn_idx = tree.query(xyz[unlabeled_mask], k=k)
    if k == 1:
        inferred_labels = assigned_labels[nn_idx]
    else:
        neighbor_labels = assigned_labels[nn_idx]
        if neighbor_labels.ndim == 1:
            neighbor_labels = neighbor_labels[None, :]
        mode_result = mode(neighbor_labels, axis=1, keepdims=False)
        inferred_labels = (
            np.asarray(mode_result.mode)
            if hasattr(mode_result, "mode")
            else np.asarray(mode_result)
        )
        if inferred_labels.ndim > 1:
            inferred_labels = inferred_labels.squeeze(-1)
        inferred_labels = inferred_labels.astype(np.int32, copy=False)
    inst_labels[unlabeled_mask] = inferred_labels

    colors_knn = colors.copy()
    colors_knn[unlabeled_mask] = inst_colors[inst_labels[unlabeled_mask]]
    pcld_knn = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcld_knn.colors = o3d.utility.Vector3dVector(colors_knn)
    knn_path = save_dir / knn_filename
    o3d.io.write_point_cloud(str(knn_path), pcld_knn)
    print(f"[Export] Saved KNN-filled colored point cloud to {knn_path}")
    knn_label_path = save_dir / "instance_labels_knn.npy"
    np.save(knn_label_path, inst_labels.copy())
    print(f"[Export] Saved KNN labels to {knn_label_path}")
