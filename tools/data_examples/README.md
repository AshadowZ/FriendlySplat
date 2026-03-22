# Dataset Examples

This directory contains tools and examples for downloading and managing various datasets used in 3D Gaussian Splatting and other 3D reconstruction tasks. Each dataset has its own download script and documentation.

## Available Dataset Examples

### 1DL3DV Benchmark Dataset

The DL3DV Benchmark is a comprehensive dataset for evaluating 3D reconstruction methods. It contains 140 scenes covering different scene complexities including reflection, transparency, indoor/outdoor environments, etc

**Original Dataset Address:** <https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark>

#### Download Script: `dl3dv_eval_download.py`

##### Prerequisites

```bash
pip install huggingface_hub tqdm pandas
```

##### Basic Usage

The `dl3dv_eval_download.py` script allows you to download specific subsets of the DL3DV benchmark dataset. Here are the main usage options:

###### Download a Specific Scene by Hash

```bash
python3 dl3dv_eval_download.py \
  --odir ./dl3dv_benchmark \
  --subset hash \
  --only_level4 \
  --hash 032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7 \
  --key hf_XXX
```

###### Download Full Benchmark

```bash
python3 dl3dv_eval_download.py \
  --odir ./dl3dv_benchmark \
  --subset full \
  --clean_cache
```

###### Download Full Benchmark with Only 960P Resolution

```bash
python3 dl3dv_eval_download.py \
  --odir ./dl3dv_benchmark \
  --subset full \
  --only_level4 \
  --clean_cache
```

##### Command Line Arguments

| Argument        | Description                                              | Required                    |
| --------------- | -------------------------------------------------------- | --------------------------- |
| `--odir`        | Local save directory                                     | No (default: DL3DV-Data)    |
| `--subset`      | Download subset: 'full' or 'hash'                        | Yes                         |
| `--only_level4` | Only download images\_4 low resolution version (960x540) | No                          |
| `--clean_cache` | Automatically clean cache after download to save space   | No                          |
| `--hash`        | Specific hash value when subset is 'hash'                | Required when subset='hash' |
| `--key`         | Hugging Face API token                                   | Yes                         |

##### Data Structure

After downloading, the dataset will be organized in the following structure:

```
dl3dv_benchmark/
├── benchmark-meta.csv        # Metadata for all scenes
└── [scene_hash]/             # Directory for each scene
    └── nerfstudio/
        ├── transforms.json   # Camera parameters and image paths
        └── images_4/         # 960x540 resolution images (if --only_level4 is used)
            ├── 000000.jpg
            ├── 000001.jpg
            └── ...
```

##### Features of the DL3DV Download Script

- **Hugging Face API Token Support**: Prevents download failures by using authenticated requests
- **Resume Download Functionality**: Automatically retries failed downloads up to 5 times
- **Error Handling and Logging**: Provides clear error messages and progress updates
- **Flexible Download Options**: Allows downloading full dataset or specific scenes
- **Space Optimization**: Option to clean cache after download to save disk space

##### Notes

- The full benchmark dataset is approximately 2.1TB, so consider using `--only_level4` to download only the 960P resolution images (around 50GB) if you have limited storage.
- You need to accept the terms and conditions on the Hugging Face dataset page before downloading.
- It's recommended to use a Hugging Face API token (--key) to avoid rate limits and ensure reliable downloads.
- If you encounter network issues when downloading, you can use a mirror by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`

## Adding New Dataset Examples

To add a new dataset example, follow these steps:

1. Create a new download script for the dataset (e.g., `new_dataset_download.py`)
2. Add documentation for the dataset in this README.md file
3. Include prerequisites, usage instructions, and data structure information
4. Ensure the script follows the same coding style and best practices as existing scripts

## Directory Structure

```
tools/data_examples/
├── README.md                  # This file - general dataset examples documentation
├── __init__.py                # Makes data_examples a proper Python package
├── dl3dv_eval_download.py     # DL3DV dataset download script
└── [future_dataset_download.py]  # Future dataset download scripts
```

