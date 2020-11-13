from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import geopandas as gpd
import multiprocessing
import pandas as pd
import numpy as np
import skimage.io
from tqdm import tqdm
import glob
import math
import gdal
import time
import sys
import os
import concurrent.futures
import skimage.io as sio
# matplotlib.use('Agg') # non-interactive
sys.path.insert(0, 'solaris')
sol = __import__('solaris')

from solaris.utils.core import _check_gdf_load
from solaris.raster.image import create_multiband_geotiff 

config = sol.utils.config.parse('yml/sn7_hrnet_infer.yml')
pred_top_dir = '/'.join(config['inference']['output_dir'].split('/')[:-1])
masks_dir = os.path.join(pred_top_dir, 'grouped')


def listdirfull(path):
  return sorted([os.path.join(path, d) for d in os.listdir(path)])

def read_data_cube(img_dir, type="tif", return_names=False):
    img_paths = [f for f in listdirfull(img_dir) if f.endswith('.tif')]
    imgs = np.stack([sio.imread(f) for f in img_paths], axis=0)
    if return_names:
        filenames = [p.split('/')[-1] for p in img_paths]
        return filenames, imgs
    else:
        return imgs

def dir_to_poly(mask_dir):
    img_dir = os.path.join(mask_dir, 'masks')
    filenames, imgs = read_data_cube(img_dir, return_names=True)
    
    # Saving directory
    out_dir = os.path.join(mask_dir, 'tracked_geojson')
    os.makedirs(out_dir, exist_ok=True)

    # Saving 
    for filename, img in zip(filenames, imgs):
        # Get geojson
        file_path = os.path.join(out_dir, filename)[:-4] + '.geojson'
        # Save to geojson
        img = img.astype('int32')
        sol.vector.mask.label_to_poly_geojson(img, output_path=file_path, 
                                                connectivity=4, min_area=3.5)

with concurrent.futures.ProcessPoolExecutor() as executor:
    results = []
    for mask_dir in listdirfull(masks_dir):
        results.append(executor.submit(dir_to_poly, mask_dir))
    
    for f in tqdm(concurrent.futures.as_completed(results)):
        pass #print(f.result())


# Save submission
def sn7_convert_geojsons_to_csv(json_dirs, output_csv_path, population='proposal'):
    '''
    Convert jsons to csv
    Population is either "ground" or "proposal" 
    '''
    
    first_file = True  # switch that will be turned off once we process the first file
    for json_dir in tqdm(json_dirs):
        json_files = sorted(glob.glob(os.path.join(json_dir, '*.geojson')))
        for json_file in tqdm(json_files):
            try:
                df = gpd.read_file(json_file)
            except (fiona.errors.DriverError):
                message = '! Invalid dataframe for %s' % json_file
                print(message)
                continue
                #raise Exception(message)
            if population == 'ground':
                file_name_col = df.image_fname.apply(lambda x: os.path.splitext(x)[0])
            elif population == 'proposal':
                file_name_col = os.path.splitext(os.path.basename(json_file))[0]
            else:
                raise Exception('! Invalid population')

            df = df.sort_values('area', ascending=False)
            df = df.drop_duplicates('Id')

            df = gpd.GeoDataFrame({
                'filename': file_name_col,
                'id': df.Id.astype(int),
                'geometry': df.geometry,
            })
            if len(df) == 0:
                message = '! Empty dataframe for %s' % json_file
                print(message)
                #raise Exception(message)
            
            if first_file:
                net_df = df
                first_file = False
            else:
                net_df = net_df.append(df)
    
    
            
    net_df.to_csv(output_csv_path, index=False)
    return net_df


out_csv_path = sys.argv[1]
prop_file = out_csv_path

final_out_dir = '/'.join(prop_file.split('/')[:-1])
if final_out_dir != '':
    os.makedirs(final_out_dir, exist_ok=True)

aoi_dirs = sorted([os.path.join(masks_dir, aoi, 'tracked_geojson') \
                   for aoi in os.listdir(os.path.join(masks_dir)) \
                   if os.path.isdir(os.path.join(masks_dir, aoi, 'tracked_geojson'))])
print("aoi_dirs:", aoi_dirs)

net_df = sn7_convert_geojsons_to_csv(aoi_dirs, prop_file, 'proposal')

print("prop_file:", prop_file)