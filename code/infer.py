import multiprocessing
import pandas as pd
import numpy as np
import skimage
import gdal
import sys
import os
from sn7.sn7_baseline_prep_funcs import map_wrapper, make_geojsons_and_masks
sys.path.insert(0, 'solaris')
sol = __import__('solaris')

# Dataset location (edit as needed)
root_dir = sys.argv[1]

# %% ============================
# Data Prep
# ===============================

# # Make dataframe csvs for test

out_dir = 'csvs/'
os.makedirs(out_dir, exist_ok=True)

d = root_dir
outpath = os.path.join(out_dir, 'sn7_baseline_test_public_df.csv')
im_list, mask_list = [], []
subdirs = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
for subdir in subdirs:
    im_files = [os.path.join(d, subdir, 'images_masked', f)
            for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
            if f.endswith('.tif')]
    im_list.extend(im_files)

# save to dataframes
# print("im_list:", im_list)
# print("mask_list:", mask_list)
df = pd.DataFrame({'image': im_list})
df.to_csv(outpath, index=False)
print("test_public len df:", len(df))
print("output csv:", outpath)


config = sol.utils.config.parse('yml/sn7_hrnet_infer.yml')
print('Config:')
print(config)

# make infernce output dir
os.makedirs(os.path.dirname(config['inference']['output_dir']), exist_ok=True)

inferer = sol.nets.infer.Inferer(config)
inferer()

pred_top_dir = '/'.join(config['inference']['output_dir'].split('/')[:-1])

raw_name = 'raw/'
grouped_name = 'grouped/'
im_list = sorted([z for z in os.listdir(os.path.join(pred_top_dir, raw_name)) if z.endswith('.tif')])
df = pd.DataFrame({'image': im_list})
roots = [z.split('mosaic_')[-1].split('.tif')[0] for z in df['image'].values]
df['root'] = roots
# copy files
for idx, row in df.iterrows():
    in_path_tmp = os.path.join(pred_top_dir, raw_name, row['image'])
    out_dir_tmp = os.path.join(pred_top_dir, grouped_name, row['root'], 'masks')
    os.makedirs(out_dir_tmp, exist_ok=True)
    cmd = 'cp ' + in_path_tmp + ' ' + out_dir_tmp
    print("cmd:", cmd)
    os.system(cmd)    



