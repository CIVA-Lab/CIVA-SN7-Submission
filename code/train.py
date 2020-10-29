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

# Load config
config_path = 'yml/sn7_hrnet_train.yml'
config = sol.utils.config.parse(config_path)

# %% ============================
# Data Prep
# ===============================

# Create Training Masks
# Multi-thread to increase speed
# We'll only make a 1-channel mask for now, but Solaris supports a multi-channel mask as well, see
#     https://github.com/CosmiQ/solaris/blob/master/docs/tutorials/notebooks/api_masks_tutorial.ipynb

aois = sorted([f for f in os.listdir(os.path.join(root_dir))
               if os.path.isdir(os.path.join(root_dir, f))])
n_threads = 10
params = [] 
make_fbc = False

input_args = []
for i, aoi in enumerate(aois):
    print(i, "aoi:", aoi)
    im_dir = os.path.join(root_dir, aoi, 'images_masked/')
    json_dir = os.path.join(root_dir, aoi, 'labels_match/')
    out_dir_mask = os.path.join(root_dir, aoi, 'masks/')
    out_dir_mask_fbc = os.path.join(root_dir, aoi, 'masks_fbc/')
    os.makedirs(out_dir_mask, exist_ok=True)
    if make_fbc:
        os.makedirs(out_dir_mask_fbc, exist_ok=True)

    json_files = sorted([f
                for f in os.listdir(os.path.join(json_dir))
                if f.endswith('Buildings.geojson') and os.path.exists(os.path.join(json_dir, f))])
    for j, f in enumerate(json_files):
        # print(i, j, f)
        name_root = f.split('.')[0]
        json_path = os.path.join(json_dir, f)
        image_path = os.path.join(im_dir, name_root + '.tif').replace('labels', 'images').replace('_Buildings', '')
        output_path_mask = os.path.join(out_dir_mask, name_root + '.tif')
        if make_fbc:
            output_path_mask_fbc = os.path.join(out_dir_mask_fbc, name_root + '.tif')
        else:
            output_path_mask_fbc = None
            
        if (os.path.exists(output_path_mask)):
             continue
        else: 
            input_args.append([make_geojsons_and_masks, 
                               name_root, image_path, json_path,
                               output_path_mask, output_path_mask_fbc])

# execute 
print("len input_args", len(input_args))
print("Execute...\n")
with multiprocessing.Pool(n_threads) as pool:
    pool.map(map_wrapper, input_args)



# Make dataframe csvs for train/test
out_path = config['training_data_csv']
os.makedirs('/'.join(out_path.split('/')[:-1]), exist_ok=True)

d = root_dir
outpath = os.path.join(out_path)
im_list, mask_list = [], []
subdirs = sorted([f for f in os.listdir(d) if os.path.isdir(os.path.join(d, f))])
for subdir in subdirs:
    
    im_files = [os.path.join(d, subdir, 'images_masked', f)
            for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
            if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
    mask_files = [os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif')
                for f in sorted(os.listdir(os.path.join(d, subdir, 'images_masked')))
                if f.endswith('.tif') and os.path.exists(os.path.join(d, subdir, 'masks', f.split('.')[0] + '_Buildings.tif'))]
    im_list.extend(im_files)
    mask_list.extend(mask_files)

df = pd.DataFrame({'image': im_list, 'label': mask_list})
df.to_csv(outpath, index=False)
print("train len df:", len(df))
print("output csv:", outpath)


# %% ============================
# Training
# ===============================

# make model output dir
os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)

trainer = sol.nets.train.Trainer(config=config)

trainer.train()