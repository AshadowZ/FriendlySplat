# Copied from: https://github.com/a1600012888/LaCT/blob/1f84d479fee89e34c58d74a0cf399c2b70564845/lact_nvs/data_preprocess/dl3dv_eval_download.py
""" This script is used to download the DL3DV benchmark from the huggingface repo.
    Copied from https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/raw/main/download.py.
    All copyrights belong to the original authors.

    Modified and enhanced by Renwang-Huang (RenwangWork@outlook.com):
    - Added Hugging Face API token support to prevent download failures
    - Optimized with resume download functionality
    - Improved error handling and logging
    - Added command-line argument for custom API token

    The benchmark is composed of 140 different scenes covering different scene complexities (reflection, transparency, indoor/outdoor, etc.) 

    The whole benchmark is very large: 2.1 TB. So we provide this script to download the subset of the dataset based on common needs. 


        - [x] Full benchmark downloading
            Full download can directly be done by git clone (w. lfs installed).

        - [x] scene downloading based on scene hash code  

        Option: 
        - [x] images_4 (960 x 540 resolution) level dataset (approx 50G)

"""

import os 
from os.path import join
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
import argparse
import traceback
import pickle
import shutil

# Default token
HF_TOKEN = "hf_XXX"

api = HfApi(token=HF_TOKEN)
repo_root = 'DL3DV/DL3DV-Benchmark'


def hf_download_path(repo_path: str, odir: str, max_try: int = 5):
    """ Download using hf_hub_download with Token

    :param repo_path: The path of the repo to download
    :param odir: output path 
    """	
    rel_path = os.path.relpath(repo_path, repo_root)

    counter = 0
    while True:
        if counter >= max_try:
            print(f"ERROR: Download {repo_path} failed (retried {max_try} times).")
            return False

        try:
            hf_hub_download(
                repo_id=repo_root,
                filename=rel_path,
                repo_type='dataset',
                local_dir=odir,
                cache_dir=join(odir, '.cache'),
                token=HF_TOKEN
            )
            return True

        except Exception as e:
            print(f"Error during download: {e}")
            counter += 1
            print(f'Retrying {counter}...')
    


def download_by_hash(filepath_dict: dict, odir: str, hash: str, only_level4: bool):
    """ Given a hash, download the relevant data from the huggingface repo 

    :param filepath_dict: the cache dict that stores all the file relative paths 
    :param odir: the download directory 
    :param hash: the hash code for the scene 
    :param only_level4: the images_4 resolution level, if true, only the images_4 resolution level will be downloaded 
    """	
    all_files = filepath_dict[hash]
    download_files = [join(repo_root, f) for f in all_files] 

    # Only kept the nerfstudio.
    download_files = [f for f in download_files if 'nerfstudio' in f]

    # Only kept the images or transforms.json
    download_files = [f for f in download_files if 'images' in f or 'transforms.json' in f]

    if only_level4: # only download images_4 level data
        download_files = [f for f in download_files if 'images_4' in f or 'images' not in f]
    
    for f in download_files:
        if hf_download_path(f, odir) == False:
            return False

    return True
    

def download_benchmark(args):
    """ Download the benchmark based on the user inputs.

        1. download the benchmark-meta.csv
        2. based on the args, download the specific subset 
            a. full benchmark 
            b. full benchmark in images_4 resolution level 
            c. full benchmark only with nerfstudio colmaps (w.o. gaussian splatting colmaps) 
            d. specific scene based on the index in [0, 140)

    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    level4_opt = args.only_level4
    hash_name  = args.hash
    is_clean_cache = args.clean_cache
    
    # Use user-provided token if available
    global HF_TOKEN
    if args.key:
        HF_TOKEN = args.key
        global api
        api = HfApi(token=HF_TOKEN)

    # import pdb; pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: download the benchmark-meta.csv and .cache/filelist.bin
    meta_repo_path = join(repo_root, 'benchmark-meta.csv')
    cache_file_path = join(repo_root, '.cache/filelist.bin')
    if hf_download_path(meta_repo_path, output_dir) == False:
        print('ERROR: Download benchmark-meta.csv failed.')
        return False

    if hf_download_path(cache_file_path, output_dir) == False:
        print('ERROR: Download .cache/filelist.bin failed.')
        return False


    # STEP 2: download the specific subset
    df = pd.read_csv(join(output_dir, 'benchmark-meta.csv'))
    filepath_dict = pickle.load(open(join(output_dir, '.cache/filelist.bin'), 'rb'))
    hashlist = df['hash'].tolist()
    download_list = hashlist

    # sanity check 
    if subset_opt == 'hash':  
        if hash_name not in hashlist: 
            print(f'ERROR: hash {hash_name} not in the benchmark-meta.csv')
            return False

        # if subset is hash, only download the specific hash
        download_list = [hash_name]

    
    # download the dataset 
    for cur_hash in tqdm(download_list):
        if download_by_hash(filepath_dict, output_dir, cur_hash, level4_opt) == False:
            return False

        if is_clean_cache:
            shutil.rmtree(join(output_dir, '.cache'))

    return True 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DL3DV Dataset Download Script')
    parser.add_argument('--odir', type=str, default='DL3DV-Data', help='Local save directory')
    parser.add_argument('--subset', choices=['full', 'hash'], required=True, help='Download full benchmark or specific hash scene')
    parser.add_argument('--only_level4', action='store_true', help='Only download images_4 low resolution version')
    parser.add_argument('--clean_cache', action='store_true', help='Automatically clean cache after download to save space')
    parser.add_argument('--hash', type=str, default='', help='Specific hash value when subset is hash')
    parser.add_argument('--key', type=str, default='', help='Hugging Face API token')
    params = parser.parse_args()

    if download_benchmark(params):
        print('\n✅ Download completed successfully! Files saved in:', params.odir)
    else:
        print('\n❌ Download failed, please check network or token permissions.')
        