from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Union

import torch
from typing_extensions import Literal


@dataclass(frozen=True)
class PreparedBatch:
    """Trainer-facing batch with normalized/typed tensors."""

    pixels: torch.Tensor
    camtoworlds: torch.Tensor
    camtoworlds_input: torch.Tensor
    Ks: torch.Tensor
    height: int
    width: int
    image_ids: Optional[torch.Tensor]
    depth_prior: Optional[torch.Tensor]
    normal_prior: Optional[torch.Tensor]
    dynamic_mask: Optional[torch.Tensor]
    sky_mask: Optional[torch.Tensor]


def prepare_batch(
    *,
    batch: Dict[str, Any],
    device: Union[str, torch.device],
) -> PreparedBatch:
    """Convert a raw dataset batch dict into `PreparedBatch`.

    This function accepts tensors already on target device (prefetch path) or CPU
    tensors (non-prefetch path), then normalizes/reshapes them into trainer format.
    """
    device_t = torch.device(device)

    def _move_if_needed(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device == device_t:
            return tensor
        return tensor.to(device_t, non_blocking=True)

    image_u8 = batch["image_u8"]
    if not isinstance(image_u8, torch.Tensor):
        raise TypeError(
            f"batch['image_u8'] must be a torch.Tensor, got {type(image_u8)}"
        )
    if image_u8.dim() != 4 or image_u8.shape[-1] != 3:
        raise ValueError(
            f"batch['image_u8'] must have shape [B,H,W,3], got {tuple(image_u8.shape)}"
        )

    camtoworlds = batch["camtoworld"]
    Ks = batch["K"]
    if not isinstance(camtoworlds, torch.Tensor) or not isinstance(Ks, torch.Tensor):
        raise TypeError("batch['camtoworld'] and batch['K'] must be torch.Tensors.")

    height, width = int(image_u8.shape[1]), int(image_u8.shape[2])

    image_u8 = _move_if_needed(image_u8)
    # RGB uint8 -> float32 in [0, 1].
    pixels = image_u8.float().div_(255.0)

    camtoworlds = _move_if_needed(camtoworlds)
    Ks = _move_if_needed(Ks)

    image_ids = batch.get("image_id")
    if isinstance(image_ids, torch.Tensor):
        image_ids = _move_if_needed(image_ids).long()
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
    # Keep the unadjusted input poses for diagnostics/compatibility.
    camtoworlds_input = camtoworlds

    depth_prior = batch.get("depth_prior_f32")
    if isinstance(depth_prior, torch.Tensor):
        depth_prior = _move_if_needed(depth_prior)

    normal_prior = None
    normal_prior_u8 = batch.get("normal_prior_u8")
    if isinstance(normal_prior_u8, torch.Tensor) and normal_prior_u8.numel() > 0:
        normal_prior_u8 = _move_if_needed(normal_prior_u8)
        # Normal maps are in OpenCV camera coordinates, but for better visualization
        # they are intentionally stored reversed, so decoding must use the inverse mapping.
        normal_prior = 1.0 - (normal_prior_u8.float() / 255.0) * 2.0

    dynamic_mask = batch.get("dynamic_mask_bool")
    if isinstance(dynamic_mask, torch.Tensor):
        dynamic_mask = _move_if_needed(dynamic_mask).bool()
        # Normalize to [B, H, W] bool for downstream loss code.
        if dynamic_mask.dim() == 4 and dynamic_mask.shape[-1] == 1:
            dynamic_mask = dynamic_mask[..., 0]
        if dynamic_mask.dim() == 2:
            dynamic_mask = dynamic_mask.unsqueeze(0)

    sky_mask = batch.get("sky_mask_bool")
    if isinstance(sky_mask, torch.Tensor):
        sky_mask = _move_if_needed(sky_mask).bool()
        # Normalize to [B, H, W] bool for downstream loss code.
        if sky_mask.dim() == 4 and sky_mask.shape[-1] == 1:
            sky_mask = sky_mask[..., 0]
        if sky_mask.dim() == 2:
            sky_mask = sky_mask.unsqueeze(0)

    return PreparedBatch(
        pixels=pixels,
        camtoworlds=camtoworlds,
        camtoworlds_input=camtoworlds_input,
        Ks=Ks,
        height=height,
        width=width,
        image_ids=image_ids,
        depth_prior=depth_prior,
        normal_prior=normal_prior,
        dynamic_mask=dynamic_mask,
        sky_mask=sky_mask,
    )


def _record_stream_recursive(obj: object, stream: torch.cuda.Stream) -> None:
    # Propagate CUDA stream usage to nested tensors to avoid premature reuse/free.
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        obj.record_stream(stream)
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _record_stream_recursive(v, stream)
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            _record_stream_recursive(v, stream)
        return


def _to_device_recursive(
    obj: object,
    device: Union[str, torch.device],
    *,
    non_blocking: bool,
) -> object:
    if isinstance(obj, torch.Tensor):
        if obj.device == torch.device(device):
            return obj
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {
            k: _to_device_recursive(v, device, non_blocking=non_blocking)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_to_device_recursive(v, device, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(
            _to_device_recursive(v, device, non_blocking=non_blocking) for v in obj
        )
    return obj


class _InfiniteRandomSampler(torch.utils.data.Sampler[int]):
    """Yield shuffled indices forever.

    This avoids DataLoader epoch boundaries (StopIteration), which can otherwise cause
    stalls while the iterator/workers refill the prefetch queue.
    """

    def __init__(self, data_source: torch.utils.data.Dataset, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        n = len(self.data_source)
        while True:
            yield from torch.randperm(n, generator=g).tolist()

    def __len__(self) -> int:
        return 2**31


class _CudaPrefetcher:
    """Prefetch DataLoader batches to GPU on a dedicated CUDA stream."""

    def __init__(
        self, loader_iter: Iterator[object], device: Union[str, torch.device]
    ) -> None:
        self.loader_iter = loader_iter
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                f"_CudaPrefetcher requires a CUDA device, got {self.device}"
            )
        self.stream = torch.cuda.Stream(device=self.device)
        self._next_batch: Optional[object] = None
        self._preload()

    def _preload(self) -> None:
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self._next_batch = None
            return
        with torch.cuda.stream(self.stream):
            # Move on prefetch stream so copy can overlap compute on default stream.
            batch = _to_device_recursive(batch, self.device, non_blocking=True)
        self._next_batch = batch

    def next(self) -> Optional[object]:
        if self._next_batch is None:
            return None
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next_batch
        _record_stream_recursive(batch, torch.cuda.current_stream(self.device))
        self._preload()
        return batch


class _PreloadedDataset(torch.utils.data.Dataset):
    """A dataset wrapper that caches samples on a target device."""

    def __init__(self, dataset: torch.utils.data.Dataset, device: torch.device) -> None:
        self._len = int(len(dataset))
        self._items: list[object] = [None] * self._len
        for i in range(self._len):
            item = dataset[i]
            self._items[i] = _to_device_recursive(item, device, non_blocking=False)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> object:
        return self._items[int(idx)]


class DataLoader:
    """A thin wrapper around torch DataLoader for gsplat trainers.

    Design goals:
      - No shuffle option; randomness is provided only via `infinite_sampler`.
      - Optional CUDA prefetching to hide H2D latency.
      - Iterable that yields `PreparedBatch` for direct trainer consumption.
      - Iterable can be used in a long training loop without manual StopIteration handling.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        persistent_workers: Optional[bool] = None,
        device: Union[str, torch.device] = "cuda",
        preload: Literal["none", "cuda"] = "none",
        infinite_sampler: bool = False,
        prefetch_to_gpu: bool = False,
        seed: int = 42,
    ):
        self.batch_size = int(batch_size)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.device = torch.device(device)
        self.preload = str(preload)
        self.infinite_sampler = bool(infinite_sampler)
        self.prefetch_to_gpu = bool(prefetch_to_gpu)
        self.seed = int(seed)
        if self.preload not in ("none", "cuda"):
            raise ValueError(f"preload must be 'none' or 'cuda', got {self.preload!r}.")
        device = self.device

        # Optional dataset-wide preloading to CUDA is owned by the loader, not the dataset.
        if self.preload == "cuda":
            assert (
                device.type == "cuda"
            ), f"preload='cuda' requires a CUDA device, got {device}."

            # When the dataset is already on CUDA, "prefetch to GPU" becomes a no-op
            # with extra stream sync overhead. Ignore it to keep behavior stable.
            if self.prefetch_to_gpu:
                print(
                    "warning: preload='cuda' ignores prefetch_to_gpu=True (data is already on CUDA).",
                    flush=True,
                )
                self.prefetch_to_gpu = False

            # Worker processes cannot safely handle CUDA tensors; force single-process.
            if self.num_workers not in (None, 0):
                print(
                    f"warning: preload='cuda' forces num_workers=0 (got {self.num_workers}).",
                    flush=True,
                )
                self.num_workers = 0

            # pin_memory is only useful for CPU->CUDA transfers; disable it for preload.
            if bool(self.pin_memory):
                print(
                    "warning: preload='cuda' forces pin_memory=False (data is already on CUDA).",
                    flush=True,
                )
                self.pin_memory = False

            # persistent_workers requires num_workers > 0; disable for preload.
            if bool(self.persistent_workers):
                print(
                    "warning: preload='cuda' forces persistent_workers=False (num_workers=0).",
                    flush=True,
                )
                self.persistent_workers = False

            dataset = _PreloadedDataset(dataset, device=device)

        self.dataset = dataset

        pin_memory = self.pin_memory
        if pin_memory is None:
            # Pin memory only helps CPU->CUDA transfers.
            pin_memory = device.type == "cuda" and self.preload != "cuda"

        num_workers = self.num_workers
        if num_workers is None:
            num_workers = 0 if self.preload == "cuda" else 8

        persistent_workers = self.persistent_workers
        if persistent_workers is None:
            persistent_workers = num_workers > 0

        if self.infinite_sampler:
            sampler = _InfiniteRandomSampler(dataset, seed=self.seed)
        else:
            sampler = None

        self._torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
        )

    def _iterate(self, *, restart_on_end: bool) -> Iterator[PreparedBatch]:
        device = self.device
        use_prefetch = (
            self.prefetch_to_gpu
            and device.type == "cuda"
            and self.preload != "cuda"
            and torch.cuda.is_available()
        )

        loader_iter = iter(self._torch_loader)
        prefetcher = _CudaPrefetcher(loader_iter, device) if use_prefetch else None
        batch = prefetcher.next() if prefetcher is not None else None

        while True:
            if prefetcher is None:
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    if not restart_on_end:
                        return
                    # Infinite iteration: restart underlying torch DataLoader.
                    loader_iter = iter(self._torch_loader)
                    batch = next(loader_iter)
            else:
                if batch is None:
                    if not restart_on_end:
                        return
                    loader_iter = iter(self._torch_loader)
                    prefetcher = _CudaPrefetcher(loader_iter, device)
                    batch = prefetcher.next()
                assert batch is not None

            # Always expose a trainer-ready structured batch.
            yield prepare_batch(batch=batch, device=device)  # type: ignore[arg-type]

            if prefetcher is not None:
                batch = prefetcher.next()

    def __iter__(self) -> Iterator[PreparedBatch]:
        return self._iterate(restart_on_end=True)

    def iter_once(self) -> Iterator[PreparedBatch]:
        """Iterate over the underlying torch DataLoader exactly once."""
        return self._iterate(restart_on_end=False)
