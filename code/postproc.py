import skimage.io as sio
from scipy import ndimage
from skimage.morphology import extrema, h_maxima, reconstruction, local_maxima, thin
from skimage.measure import label
from skimage.segmentation import watershed
from skimage.morphology import thin

import cv2
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.insert(0, 'solaris')
sol = __import__('solaris')


def listdirfull(path):
    return sorted([os.path.join(path, d) for d in os.listdir(path)])


def to_uint8(mask):
    """
      mask: mask or heatmap
    """
    return (255 * (mask > 0)).astype('uint8')


def persistence(unet_output):

    # unet_output: a list of masks: unet distance + hyThresh + median_filter5 + persistence output masks
    unet_output_np = np.asarray(unet_output)

    h = 2
    mask_thresh = 0

    sum_img = []
    for i, img in enumerate(unet_output_np):
        img = img > 0
        img = img.astype(int)
        if i == 0:
            sum_img = img
        else:
            sum_img = sum_img + img

    h_maxima_output = reconstruction(
        sum_img-h, sum_img, method='dilation', selem=np.ones((3, 3), dtype=int), offset=None)
    region_max = local_maxima(h_maxima_output, connectivity=2)
    label_h_maxima = label(region_max, connectivity=2)
    # use peaks and summed images to get watershed separation line
    labels = watershed(-sum_img, label_h_maxima,
                       watershed_line=True, connectivity=2)
    split_line = labels == 0
    split_line = split_line.astype(int)

    # split_line = thin(split_line)

    split_line = np.where(sum_img == 0, 0, split_line)

    new_img = []
    for i, img in enumerate(unet_output_np):
        split_img = img > 0
        split_img = split_img.astype(int)
        split_img = np.where(split_line == 1, 0, split_img)
        split_img = split_img * 255
        new_img.append(split_img)

    return new_img


config = sol.utils.config.parse('yml/sn7_hrnet_infer.yml')
pred_top_dir = '/'.join(config['inference']['output_dir'].split('/')[:-1])
test_data_dir = sys.argv[1]

out_dirs = [os.path.join(d, 'masks')
            for d in listdirfull(pred_top_dir + '/grouped')]
inp_dirs = [os.path.join(d, 'images_masked')
            for d in listdirfull(test_data_dir)]


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
        masks = np.stack([sio.imread(f)[:, :, 3]/255 for f in inp_files])
        ims = ndimage.median_filter(ims, size=(k, 1, 1))
        ims = ims * masks
        # print(ims.shape)

        # preparing file names list
        file_names = [f.split('/')[-1] for f in out_files]
        ofiles = [os.path.join(out_dir, f) for f in file_names]
        print(ofiles[0])

        for im, f in zip(ims, ofiles):
            sio.imsave(f, im)


unet_persis_path = pred_top_dir + '/grouped'
save_path = pred_top_dir + '/grouped'

aois = sorted([f for f in os.listdir(os.path.join(unet_persis_path)) if os.path.isdir(
    os.path.join(unet_persis_path, f))])

for aoi in aois:
    print(aoi)
    # ============ read images ============
    unet_persis_img_path = sorted([f for f in os.listdir(os.path.join(
        unet_persis_path, aoi, 'masks')) if f.endswith('tif')])
    unet_persis_img_list = []
    for _, unet_per_img_path in enumerate(unet_persis_img_path):
        unet_per_img_path = os.path.join(
            unet_persis_path, aoi, 'masks', unet_per_img_path)
        unet_per_img = cv2.imread(unet_per_img_path, -1)
        unet_persis_img_list.append(unet_per_img)

    # ============ main function ============
    # hrnet_persis_img is one persistence image of one data cube
    # unet_persis_img_list is a list of images of that data cube
    # output [new_unet_imgs] is 0/255 masks list
    new_unet_imgs = persistence(unet_persis_img_list)

    # ============ save images ============
    for i, unet_per_img_path in enumerate(unet_persis_img_path):
        save_imgs_path = os.path.join(
            save_path, aoi, 'masks', unet_per_img_path)
        os.makedirs(os.path.join(save_path, aoi,
                                 'masks'), exist_ok=True)
        cv2.imwrite(save_imgs_path, new_unet_imgs[i].astype(np.uint8))
