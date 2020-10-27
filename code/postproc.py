import skimage.io as sio
from scipy import ndimage
import cv2
import os
import sys
from tqdm import tqdm
import numpy as np

sys.path.insert(0, 'solaris')
sol = __import__('solaris')

def listdirfull(path):
  return sorted([os.path.join(path, d) for d in os.listdir(path)])

def to_uint8(mask):
  """
    mask: mask or heatmap
  """
  return (255 * (mask > 0)).astype('uint8')

import pandas as pd

config = sol.utils.config.parse('yml/sn7_hrnet_infer.yml')
pred_top_dir = '/'.join(config['inference']['output_dir'].split('/')[:-1])
test_data_dir = sys.argv[1]

out_dirs = [os.path.join(d, 'masks') for d in listdirfull(pred_top_dir + '/grouped')] 
inp_dirs = [os.path.join(d, 'images_masked') for d in listdirfull(test_data_dir)] 


out_files_list = [listdirfull(d) for d in out_dirs]
inp_files_list = [listdirfull(d) for d in inp_dirs]

for inp_files, out_files in zip(inp_files_list, out_files_list):
    for k in [5]:
        # Make directory if not exists
        dir_parts = out_files[0].split('/')[:-2]
        # dir_parts[-3] = f'test-hrnet-preds-median-{k}'
        out_dir = '/'.join([*dir_parts, 'masks'])
        os.makedirs(out_dir, exist_ok=True)

        ims = np.stack([sio.imread(f) for f in out_files])
        masks = np.stack([sio.imread(f)[:,:,3]/255 for f in inp_files])
        ims = ndimage.median_filter(ims, size=(k, 1, 1))
        ims = ims * masks
        # print(ims.shape)

        
        # preparing file names list
        file_names = [f.split('/')[-1] for f in out_files]
        ofiles = [os.path.join(out_dir, f) for f in file_names]
        print(ofiles[0])

        for im, f in zip(ims, ofiles):
            sio.imsave(f, im)
