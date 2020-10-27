from osgeo import gdal
import rasterio
from affine import Affine
import numpy as np
import logging
from ..utils.raster import reorder_axes
from ..utils.log import _get_logging_level
import os
import cv2
from scipy import ndimage


def get_geo_transform(raster_src):
    """Get the geotransform for a raster image source.

    Arguments
    ---------
    raster_src : str, :class:`rasterio.DatasetReader`, or `osgeo.gdal.Dataset`
        Path to a raster image with georeferencing data to apply to `geom`.
        Alternatively, an opened :class:`rasterio.Band` object or
        :class:`osgeo.gdal.Dataset` object can be provided. Required if not
        using `affine_obj`.

    Returns
    -------
    transform : :class:`affine.Affine`
        An affine transformation object to the image's location in its CRS.
    """

    if isinstance(raster_src, str):
        affine_obj = rasterio.open(raster_src).transform
    elif isinstance(raster_src, rasterio.DatasetReader):
        affine_obj = raster_src.transform
    elif isinstance(raster_src, gdal.Dataset):
        affine_obj = Affine.from_gdal(*raster_src.GetGeoTransform())

    return affine_obj


def stitch_images(im_arr, idx_refs=None, out_width=None,
                  out_height=None, method='average', use_GPU=True):
    """Stitch together images into a single 2- or 3-channel array.

    This function helps combine predictions generated by inferencing tiled
    pieces of larger images, similar to the pre-existing CosmiQ Works tool,
    BASISS_

    .. _BASISS: https://github.com/cosmiq/basiss

    Arguments
    ---------
    im_arr : :class:`numpy.array` or :class:`list` of :class:`numpy.array` s
        A 3- or 4-D :class:`numpy.array` with shape ``[N, Y, X(, C)]`` or a
        list of length N made up of 2- or 3-D tensors with shape
        ``[Y, X(, C)]``. These array(s) will be stitched together to produce a
        single output of shape ``[Y, X(, C)]`` .
    idx_refs : list, optional
        A list of ``(Y, X)`` indices for each sub-array to define the location
        of the first corner in the final output. Used for stitching together
        non-overlapping or partially overlapping tiles into a single output.
        Note that the index reference output of
        :class:`solaris.nets.datagen.InferenceTiler` provides the required
        reference system for stitching here.
    out_width : int, optional
        The width of the output array in pixels. If not provided, it is assumed
        that the width is the same as the width of ``im_arr`` .
    out_height : int, optional
        The height of the output array in pixels. If not provided, it is
        assumed that the height is the same as the height of ``im_arr`` .
    method : str, optional
        possible values are ``'average'``  (default), ``'first'`` , and
        ``'confidence'`` .
        * If ``'average'`` , all pixels corresponding to the same location in
        ``[Y, X, C]`` space are averaged.
        * If ``'first'`` , the value of the first pixel along the ``N`` axis
        for a given ``[Y, X, C]`` location is selected.
        * If ``'confidence'`` , it's assumed that pixel values correspond to
        probabilities in ``[0, 1]`` . In this case, for a given ``[Y, X, C]``
        location, the pixel with the greatest distance from ``0.5`` will be
        selected (being the value with the highest confidence).
    use_GPU : bool, optional
        Should processing be performed on the GPU if a GPU is available?
        Defaults to yes (``True``). If a GPU isn't available, this argument is
        ignored. ``False`` will force CPU-located processing.

    Returns
    -------
    output_arr : a :class:`numpy.array` with shape ``[Y, X(, C)]`` .
    """
    # determine what shape the input is and stitch together accordingly
    if isinstance(im_arr, list):
        im_arr = np.stack(im_arr)  # stack along a new 1st axis

    im_arr = reorder_axes(im_arr, 'tensorflow')

    if idx_refs is not None:
        if len(idx_refs) != im_arr.shape[0]:
            raise ValueError('len(idx_refs) must be equal to the number of '
                             'images being stitched.')
    if idx_refs is not None and (out_width is None or out_height is None):
        raise ValueError('If idx_refs are provided, the desired '
                         'out_height and out_width must be provided as well.')
    if len(im_arr.shape) == 4:
        has_channels = True
    elif len(im_arr.shape) == 3:
        has_channels = False

    if idx_refs is not None:  # proxy for whether dims were provided as args
        if has_channels:
            stitching_arr = np.empty(shape=(im_arr.shape[0],
                                            out_height, out_width,
                                            im_arr.shape[3]))
        else:
            stitching_arr = np.empty(shape=(im_arr.shape[0],
                                            out_height, out_width))
        stitching_arr[:] = np.nan
        for idx in range(len(idx_refs)):
            if has_channels:
                stitching_arr[
                    idx,
                    idx_refs[idx][0]:idx_refs[idx][0] + im_arr.shape[1],
                    idx_refs[idx][1]:idx_refs[idx][1] + im_arr.shape[2],
                    :] = im_arr[idx, :, :, :]
            else:
                stitching_arr[
                    idx,
                    idx_refs[idx][0]:idx_refs[idx][0] + im_arr.shape[1],
                    idx_refs[idx][1]:idx_refs[idx][1] + im_arr.shape[2]] = im_arr[idx, :, :]
    else:
        stitching_arr = im_arr  # just stitching across images with no offset

    if method == 'average':
        output_arr = np.nanmean(stitching_arr, axis=0)

    elif method == 'first':
        # get index along 1st axis of the first non-NaN value
        first_non_nan = np.invert(np.isnan(stitching_arr)).argmax(axis=0)
        # subset along 1st axis for only the first non-NaN value
        output_arr = np.take_along_axis(stitching_arr,
                                        np.expand_dims(first_non_nan, axis=0),
                                        axis=0)[0, :, :, :]  # drop extra axis

    elif method == 'confidence':
        # convert from 0-1 to 0-0.5, values originally 0.5 become 0
        conf_scale = np.abs(stitching_arr - 0.5)
        # set NaN values to -1 so they're never selected
        conf_scale[np.isnan(conf_scale)] = -1
        # get highest conf slice at each [Y, X, C] position
        max_conf_ind = conf_scale.argmax(axis=0)
        # subset to take only the highest-conf value
        output_arr = np.take_along_axis(stitching_arr,
                                        np.expand_dims(max_conf_ind, axis=0),
                                        axis=0)[0, :, :, :]  # drop extra axis
    output_arr = output_arr.astype(im_arr.dtype)

    return output_arr


def create_multiband_geotiff(array, out_name, proj, geo, nodata=0,
                             out_format=gdal.GDT_Byte, verbose=False):
    """Convert an array to an output georegistered geotiff.
    Arguments
    ---------
    array  : :class:`numpy.ndarray`
        A numpy array with a the shape: [Channels, X, Y] or [X, Y]
    out_name : str
        The output name and path for your image
    proj : :class:`gdal.projection`
        A projection, can be extracted from an image opened with gdal with
        image.GetProjection(). Can be set to None if no georeferencing is
        required.
    geo : :class:`gdal.geotransform`
        A gdal geotransform which indicates the position of the image on the
        earth in projection units. Can be set to None if no georeferencing is
        required. Can be extracted from an image opened with gdal with
        image.GetGeoTransform()
    nodata : int
        A value to set transparent for GIS systems.
        Can be set to None if the nodata value is not required. Defaults to 0.
    out_format : str, gdalconst
        https://gdal.org/python/osgeo.gdalconst-module.html
        Must be one of the variables listed in the docs above. Defaults to
        gdal.GDT_Byte.
    verbose : bool
        A verbose output, printing all inputs and outputs to the function.
        Useful for debugging. Default to `False`
    """
    driver = gdal.GetDriverByName('GTiff')
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)
    dataset = driver.Create(
        out_name, array.shape[2], array.shape[1], array.shape[0], out_format)
    if verbose is True:
        print("Array Shape, should be [Channels, X, Y] or [X,Y]:", array.shape)
        print("Output Name:", out_name)
        print("Projection:", proj)
        print("GeoTransform:", geo)
        print("NoData Value:", nodata)
        print("Bit Depth:", out_format)
    if proj is not None:
        dataset.SetProjection(proj)
    if geo is not None:
        dataset.SetGeoTransform(geo)
    if nodata is None:
        for i, image in enumerate(array, 1):
            dataset.GetRasterBand(i).WriteArray(image)
        del dataset
    else:
        for i, image in enumerate(array, 1):
            dataset.GetRasterBand(i).WriteArray(image)
            dataset.GetRasterBand(i).SetNoDataValue(nodata)
        del dataset


def get_intensity_quantiles(dataset_dir, percentiles=[0, 100], ext="tif",
                            recursive=False, channels=None, verbose=0):
    """Get approximate dataset pixel intensity percentiles for normalization.

    This function reads every image in a dataset directory and gets a rough
    approximation of pixel intensity percentiles

    Arguments
    ---------
    """
    pass


class ScaleFunction:
    def __init__(self, compression_delta, **kwargs):
        self.compression_delta = compression_delta
        for k, v in kwargs.items():
            setattr(self, k, v)

    def forward(self, quantile):
        raise NotImplementedError

    def inverse(self, k):
        raise NotImplementedError


class K1ScaleFunction(ScaleFunction):
    """Calculate the k1 scale function for a quantile given a comp. factor."""

    def __init__(self, compression_delta):
        self.super().__init__(self, compression_delta)

    def forward(self, quantile):
        return (self.compression_delta / 2 * np.pi) * np.arcsin(2 * quantile - 1)

    def inverse(self, k):
        return np.sin((2 * np.pi * k) / self.compression_delta) + 1


def get_tdigest(data_buffer, tdigest=None, scale_function=K1ScaleFunction,
                compression_delta=0.01):
    """Create a new t-digest or merge it with an existing digest.

    This function is an implementation of Algorithm 1 from https://github.com/tdunning/t-digest/blob/master/docs/t-digest-paper/histo.pdf

    Arguments
    ---------
    data_buffer : :class:`numpy.ndarray`
        An array of data to load into a tdigest. This will be flattened into
        a vector.
    tdigest : :class:`TDigest`, optional
        An existing :class:`TDigest` object to merge with the new data.
    """
    buffer_centroids = data_buffer.flatten().astype(np.float32)
    buffer_weights = np.ones(shape=buffer_centroids.shape, dtype=np.float32)
    if tdigest:
        buffer_centroids = np.concatenate((buffer_centroids,
                                           tdigest.centroids))
        buffer_weights = np.concatenate((buffer_weights, tdigest.weights))

    # sort the data buffer on values (union with tdigest centroids if given)
    sort_order = np.argsort(buffer_centroids)
    buffer_centroids = buffer_centroids[sort_order]
    buffer_weights = buffer_weights[sort_order]
    S = buffer_weights.sum()
    out_centroids = np.array([], dtype=np.float32)
    out_weights = np.array([], dtype=np.float32)
    q0 = 0.
    q_limit = _get_q_limit(scale_function, q0, compression_delta)
    sigma_centroid = buffer_centroids[0]
    sigma_weight = buffer_weights[0]
    for idx in range(1, len(buffer_centroids)):
        q = q0 + (sigma_weight + buffer_weights[idx]) / S
        if q <= q_limit:
            sigma_centroid = ((sigma_centroid * sigma_weight
                               + buffer_centroids[idx] * buffer_weights[idx]) /
                              (sigma_weight + buffer_weights[idx]))
            sigma_weight = sigma_weight + buffer_weights[idx]
        else:
            np.append(out_centroids, sigma_centroid)
            np.append(out_weights, sigma_weight)
            q0 += sigma_weight / S
            q_limit = _get_q_limit(scale_function, q0, compression_delta)
            sigma_weight = buffer_weights[idx]
            sigma_centroid = buffer_centroids[idx]
    np.append(out_centroids, sigma_centroid)
    np.append(out_weights, sigma_weight)

    return out_centroids, out_weights


def _get_q_limit(scale_function, q0, compression_delta):
    return 1. / (scale_function(scale_function(q0, compression_delta) + 1,
                                compression_delta))


def temporal_median(masks_arr, k=5):
    return ndimage.median_filter(masks_arr, size=(k, 1, 1))


def heatmap_to_mask(im):
    """ Converts logits map from a network to a binary mask using
        a hard-limiter.

    Args:
        im ((H, W) np.ndarray): input image

    Returns:
        ((H, W) np.ndarray): output mask
    """
    return ((im > 0) * 255).astype('uint8')


def heatmap_to_cc_with_areas(im, connectivity=4):
    """ Generate connected components and associated areas
        from prediction map.

    Args:
        im ((H, W) np.ndarray): input heatmap
        connectivity (int, optional): Connectivity for connected
            components . Defaults to 4.

    Returns:
        (H, W) np.ndarray: output connected components
    """
    im = heatmap_to_mask(im)
    num_labels, im_cc, stats, centroids = cv2.connectedComponentsWithStats(
        im, connectivity=connectivity)
    im_areas = stats[:, 4]
    return im_cc, im_areas


def track_two_frames(im1_cc, im1_areas, im2_cc, im2_areas, new_idx_start=None, iou_threshold=0.25, verbose=False):
    """ Helper function for `track_masks`. 
    Matches two adjacent frames.

    Args:
        im1_cc ((H, W) np.ndarray): labeled image at frame `t`
        im2_cc ((H, W) np.ndarray): labeled image at frame `t + 1`

    Returns:
        (H, W) np.ndarray: updated labeled image at `t + 1`
    """
    # Will hold indeces of new buildings
    idx = max(im1_cc.max(), im2_cc.max()) + \
        1 if new_idx_start is None else new_idx_start

    matches = np.stack([im1_cc, im2_cc], axis=-1).reshape(-1, 2)
    correspondences, counts = np.unique(matches, axis=0, return_counts=True)

    new_im2_cc = np.zeros_like(im2_cc)

    # # Makes sure new buildings are matched to new ids:
    for [label1, label2] in (correspondences):
        if label1 == 0 and label2 != 0:
            new_im2_cc[im2_cc == label2] = idx
            idx += 1

    # Match common buildings
    # seen_labels = []
    seen_ious_1 = {}
    seen_ious_2 = {}

    for i, [label1, label2] in enumerate(correspondences):
        # CASE (X, Y) > 0
        if label1 != 0 and label2 != 0:
            # Calculate IOU
            area1 = im1_areas[label1]
            area2 = im2_areas[label2]

            intersection = counts[i]
            union = area1 + area2 - intersection

            iou = intersection / union

            # if label1 in seen_ious_1 or label2 in seen_ious_2:
            # CASE X or Y have been seen
            if label1 in seen_ious_1 or label2 in seen_ious_2:
                # BUILDING SPLIT CASE: X has been seen
                if label1 in seen_ious_1:
                    if iou > seen_ious_1[label1]:
                        # Assign a new label to the old label2
                        new_im2_cc[new_im2_cc == label1] = idx

                        # This becomes the better label
                        new_im2_cc[im2_cc == label2] = label1

                        seen_ious_1[label1] = iou
                        seen_ious_2[label2] = iou
                    else:
                        new_im2_cc[im2_cc == label2] = idx
                    # seen_labels.append(idx)
                    idx += 1
                # BUILDING MERGE CASE: Y has been een
                elif label2 in seen_ious_2:
                    if iou > seen_ious_2[label2]:
                        # Assign a new label to the old label2
                        new_im2_cc[new_im2_cc == label1] = idx

                        # This becomes the better label
                        new_im2_cc[im2_cc == label2] = label1

                        seen_ious_1[label1] = iou
                        seen_ious_2[label2] = iou

                    # # Make previous match its original Id
                    # prev_detection_mask = (im1_cc == label1) * (new_im2_cc == label2)
                    # new_im2_cc[prev_detection_mask] = im1_cc[prev_detection_mask]

                    # idx += 1
            elif iou < iou_threshold:
                new_im2_cc[im2_cc == label2] = idx
                idx += 1
            else:
                # Update the
                new_im2_cc[im2_cc == label2] = label1

                seen_ious_1[label1] = iou
                seen_ious_2[label2] = iou

            if verbose:
                print("Pair", (label1, label2), ":",
                      new_im2_cc, seen_ious_1, seen_ious_2)

    new_areas = [(new_im2_cc == i).sum()
                 for i in np.arange(new_im2_cc.max() + 1)]

    return new_im2_cc, new_areas, idx


def track_masks(ims_arr, iou_threshold=0.25):
    """Tracks frames in the pixel level

    Args:
        ims_arr ((N, H, W) np.ndarray): Input image sequence 
            (typically the network outputs stacked together)

        output_dir ((N, H, W) np.ndarray): Output labeled image
            sequence
    """

    # Start with the first frame
    im1_cc, im1_areas = heatmap_to_cc_with_areas(ims_arr[0])

    # Save the first frame
    output = [im1_cc]

    new_idx_start = im1_cc.max() + 1

    # For each following frame
    for im in ims_arr[1:]:
        im2_cc, im2_areas = heatmap_to_cc_with_areas(im)
        im2_cc, im1_areas, new_idx_start = track_two_frames(
            output[-1], im1_areas, im2_cc, im2_areas, new_idx_start=new_idx_start, 
            iou_threshold=iou_threshold)
        output.append(im2_cc)

    return np.stack(output, axis=0)
