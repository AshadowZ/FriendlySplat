# Urban Scenes Visual + Geometry Quality Benchmark

This folder is reserved for an end-to-end benchmarking pipeline on large-scale urban
scenes (e.g. GauU-Scene / MatrixCity style datasets).

Planned stages (scripts will live here):
- Preprocess: image downsample + prior generation (e.g. MoGe normal/depth).
- Train: coarse/global model.
- Partition: split into spatial blocks and assign images per block (optional visibility/content).
- Train: per-partition fine-tuning and exports.

