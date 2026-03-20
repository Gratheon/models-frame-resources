#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:34:46 2018

@author: avsthiago
"""

import numpy as np
import cv2
import os
import h5py
import json
import gc  # Added for explicit garbage collection

# Memory optimization: Configure TensorFlow before importing
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import tensorflow.keras.backend as tb
# tb._SYMBOLIC_SCOPE.value = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import tensorflow.keras.backend as K
from tensorflow.keras.models import model_from_json, load_model
# from keras.models import load_model, model_from_json

from keras.applications.imagenet_utils import preprocess_input

import math
from collections import Counter
import datetime
import warnings
import imghdr
from pathlib import PurePath
import io # Added for in-memory buffer handling
from threading import Lock

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# address some inteeface discrepancies when using tensorflow.keras
# if "slice" not in K.__dict__ and K.backend() == "tensorflow":
#     # this is a good indicator that we are using tensorflow.keras
#
#     try:
#         # at first try to monkey patch what we need, will only work if keras-team keras is installed
# from keras import backend as KKK
import tensorflow.compat.v2 as tf


# Monkey patch func_load to handle bad marshal data from legacy Lambda layers
try:
    from keras.utils import generic_utils
    _original_func_load = generic_utils.func_load
    
    def safe_func_load(code, defaults=None, globs=None):
        try:
            return _original_func_load(code, defaults, globs)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to unmarshal lambda function, using fallback: {e}")
            # Return a function that handles the tensor splitting for classification model
            def fallback_lambda(x, i=None, parts=None, **kwargs):
                import tensorflow as tf
                # Handle the tensor splitting for multi-GPU training setup
                if i is not None and parts is not None:
                    # This Lambda was likely used for multi-GPU data splitting
                    # Since we're running on single GPU/CPU, just return the input
                    # But we need to handle the batch dimension properly
                    try:
                        batch_size = tf.shape(x)[0]
                        # Calculate the slice size for this part
                        slice_size = batch_size // parts
                        start_idx = i * slice_size
                        if i == parts - 1:  # Last part gets remainder
                            return x[start_idx:]
                        else:
                            return x[start_idx:start_idx + slice_size]
                    except:
                        # If slicing fails, return the whole tensor
                        return x
                # For other Lambda functions, return identity
                return x
            return fallback_lambda
    
    generic_utils.func_load = safe_func_load
except Exception as e:
    print(f"Warning: Could not patch func_load: {e}")

# try:
K.__dict__.update(
    is_tensor=tf.is_tensor,
    slice=tf.slice,
)
# finally:
#     del KKK

#     except ImportError:
#         # if that doesn't work we do a dirty copy of the code required
#         import tensorflow as tf
#         from tensorflow.python.framework import ops as tf_ops
#
#
#         def is_tensor(x):
#             return isinstance(x, tf_ops._TensorLike) or tf_ops.is_dense_tensor_like(x)
#
#
#         def slice(x, start, size):
#             x_shape = K.int_shape(x)
#             if (x_shape is not None) and (x_shape[0] is not None):
#                 len_start = K.int_shape(start)[0] if is_tensor(start) else len(start)
#                 len_size = K.int_shape(size)[0] if is_tensor(size) else len(size)
#                 if not (len(K.int_shape(x)) == len_start == len_size):
#                     raise ValueError('The dimension and the size of indices should match.')
#             return tf.slice(x, start, size)
        
PATH = os.path.dirname(os.path.realpath("__file__"))

ROOT = '/app'

PATH_SEG_MODEL = f'{ROOT}/src/DeepBee/software/model/segmentation.h5'
PATH_SEG_MODEL_JSON = f'{ROOT}/src/DeepBee/software/model/segmentation.model.json'
PATH_SEG_MODEL_WEIGHTS = f'{ROOT}/src/DeepBee/software/model/segmentation.weights.h5'

PATH_CL_MODEL = f'{ROOT}/src/DeepBee/software/model/classification.h5'
PATH_CL_MODEL_JSON = f'{ROOT}/src/DeepBee/software/model/classification.model.json'
PATH_CL_MODEL_WEIGHTS = f'{ROOT}/src/DeepBee/software/model/classification.weights.h5'

MIN_CONFIDENCE = 0.9995

LEFT_BAR_SIZE = 480


def _load_positive_int_env(name, default):
    raw_value = os.environ.get(name, str(default))
    try:
        parsed = int(raw_value)
        if parsed <= 0:
            raise ValueError(f"{name} must be > 0")
        return parsed
    except Exception:
        print(f"Invalid {name}={raw_value!r}; falling back to {default}")
        return default


# Keep trained input size as default for precision; make configurable for constrained runtimes.
img_size = _load_positive_int_env("CLASSIFICATION_IMG_SIZE", 224)
classification_batch_size = _load_positive_int_env("CLASSIFICATION_BATCH_SIZE", 32)

# Global model cache for lazy loading to fix idle memory usage
_segmentation_model = None
_classification_model = None
_segmentation_model_lock = Lock()
_classification_model_lock = Lock()

def load_segmentation_model():
    """Lazy load segmentation model only when needed"""
    global _segmentation_model
    if _segmentation_model is None:
        with _segmentation_model_lock:
            if _segmentation_model is None:
                print("Loading segmentation model...")
                with open(PATH_SEG_MODEL_JSON, 'r') as json_file:
                    model_json = json_file.read()
                    _segmentation_model = model_from_json(model_json, custom_objects={"K": K})
                    _segmentation_model.load_weights(PATH_SEG_MODEL_WEIGHTS)
    return _segmentation_model

def load_classification_model():
    """Lazy load classification model only when needed"""
    global _classification_model
    if _classification_model is None:
        with _classification_model_lock:
            if _classification_model is None:
                print("Loading classification model...")
                with open(PATH_CL_MODEL_JSON, 'r') as json_file:
                    model_json = json_file.read()
                    _classification_model = model_from_json(model_json, custom_objects={"K": K})
                    _classification_model.load_weights(PATH_CL_MODEL_WEIGHTS)
    return _classification_model

def unload_models():
    """Unload models from memory to reduce idle usage"""
    global _segmentation_model, _classification_model
    if _segmentation_model is not None:
        del _segmentation_model
        _segmentation_model = None
    if _classification_model is not None:
        del _classification_model
        _classification_model = None
    cleanup_memory()
    print("Models unloaded from memory")

def cleanup_memory():
    """Force garbage collection and clear Keras session"""
    K.clear_session()
    gc.collect()

def get_qtd_by_class(points, labels):
    points_filtered = points[points[:, 4] == 1, 3]
    sum_predictions = Counter(points_filtered)
    return [
        *[str(sum_predictions[i]) for i, j in enumerate(labels)],
        str(len(points_filtered)),
    ]


def get_header(labels):
    return "Img Name," + ",".join([i for i in labels]) + ",Total\n"


def draw_labels_bar(image, labels, colors):
    height = image.shape[0]
    left_panel = np.zeros((height, LEFT_BAR_SIZE, 3), dtype=np.uint8)
    labels = [l.title() for l in labels]

    for i, cl in enumerate(zip(colors, labels)):
        color, label = cl
        cv2.putText(
            left_panel,
            " ".join([str(i + 1), ".", label]),
            (15, 70 * (i + 1)),
            cv2.FONT_HERSHEY_DUPLEX,
            1.4,
            color,
            2,
        )

    return np.hstack((left_panel, image))


def draw_circles_labels(image, labels, points, colors=None, draw_labels=True):
    if colors is None:

        colors = [
            (255, 0, 0),
            (0, 255, 255),
            (0, 0, 128),
            (255, 0, 255),
            (0, 255, 0),
            (255, 255, 100),
            (0, 0, 255),
        ]

    if draw_labels:
        image = draw_labels_bar(np.copy(image), labels, colors)

    points[:, 0] += LEFT_BAR_SIZE

    for p in points:
        cv2.circle(image, (p[0], p[1]), p[2], colors[p[3]], 4)

    points[:, 0] -= LEFT_BAR_SIZE
    return image


def extract_circles(
    image, pts, output_size=224, mean_radius_default=32, standardize_radius=True
):
    """
    extract cells from a image:
    Parameters
    ----------
    image : image with full size
    pts : ndarray with a set of points in the shape [W, H, R] R stands for
          radius
    output_size : all images will be returned with the size
                  (output_size, output_size)
    mean_radius_default : if standardize_radius is set, thes parameter will be
                          used as a base size to resize all circle detections
                          32 is the average radius of a cell
    Returns
    -------
    ROIs : (N x W x H x C) N as the total number of detections and K is the
           number of channels
    """
    if standardize_radius:
        # use the mean radius to calculate the clip size to each detection
        pts[:, 2] = output_size / mean_radius_default * pts[:, 2]
        # the border needs to be greater than the biggest clip
        size_border = int(pts[:, 2].max() + 1)
        # deslocates the detection centers
        pts[:, [0, 1]] = pts[:, [0, 1]] + size_border

        # creates a border around the main image
        img_w_border = cv2.copyMakeBorder(
            image,
            size_border,
            size_border,
            size_border,
            size_border,
            cv2.BORDER_REFLECT,
        )

        # extracts all detections and resizes them
        ROIs = [
            cv2.resize(
                img_w_border[i[1] - i[2] : i[1] + i[2], i[0] - i[2] : i[0] + i[2]],
                (output_size, output_size),
            )
            for i in pts
        ]

    return ROIs


def extract_circles_batches(
    image,
    pts,
    output_size=224,
    mean_radius_default=32,
    standardize_radius=True,
    batch_size=32,
):
    """
    Yield extracted circle crops in batches to avoid building a full tensor for all cells.
    """
    if not standardize_radius or len(pts) == 0:
        return

    pts_local = np.copy(pts)
    pts_local[:, 2] = output_size / mean_radius_default * pts_local[:, 2]
    pts_local[:, 2] = np.maximum(pts_local[:, 2], 1).astype(np.int32)

    size_border = int(pts_local[:, 2].max() + 1)
    pts_local[:, [0, 1]] = pts_local[:, [0, 1]] + size_border

    img_w_border = cv2.copyMakeBorder(
        image,
        size_border,
        size_border,
        size_border,
        size_border,
        cv2.BORDER_REFLECT,
    )

    for start in range(0, len(pts_local), batch_size):
        batch_pts = pts_local[start : start + batch_size]
        batch_rois = [
            cv2.resize(
                img_w_border[p[1] - p[2] : p[1] + p[2], p[0] - p[2] : p[0] + p[2]],
                (output_size, output_size),
            )
            for p in batch_pts
        ]
        yield np.asarray(batch_rois), start, len(batch_pts)


# Corrected classify_image signature and removed obsolete file operations
def classify_image(image, points, labels, net, img_size):
    try:
        if len(points) == 0:
            return np.array([])
            
        # Convert to RGB if needed  
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        pt = np.copy(points)
        pt[:, 2] = pt[:, 2] // 2

        # Limit maximum number of cells to process at once
        max_cells = MAX_CLASSIFICATION_CELLS
        if len(pt) > max_cells:
            print(
                f"Large frame with {len(pt)} cells detected, "
                f"processing first {max_cells} (MAX_CLASSIFICATION_CELLS) for memory efficiency"
            )
            points = points[:max_cells]
            pt = pt[:max_cells]

        num_points = len(points)
        if num_points == 0:
            return np.array([])

        print(
            f"Processing {num_points} cells in streaming batches of {classification_batch_size}"
        )

        class_count = int(net.output_shape[-1])
        scores = np.zeros((num_points, class_count), dtype=np.float32)

        for batch_imgs, start, batch_len in extract_circles_batches(
            image_rgb,
            pt,
            output_size=img_size,
            batch_size=classification_batch_size,
        ):
            try:
                batch_imgs = batch_imgs.astype(np.float32)
                try:
                    batch_imgs = preprocess_input(batch_imgs)
                except Exception:
                    batch_imgs = batch_imgs / 255.0

                chunk_scores = net.predict(
                    batch_imgs,
                    verbose=0,
                    batch_size=batch_len,
                )
                scores[start : start + batch_len] = chunk_scores
            except Exception as e:
                print(f"Classification batch {start // classification_batch_size + 1} failed: {e}")
                # Keep zeroed logits for this chunk so request can complete.

        lb_predictions = np.argmax(scores, axis=1)
        vals_predictions = np.amax(scores, axis=1)

        points_pred = np.hstack((np.copy(points), np.expand_dims(lb_predictions, axis=0).T))
        new_class = np.copy(points_pred[:, 3])
        csl = np.vstack([new_class, vals_predictions]).T
        points_pred = np.hstack((points_pred, csl))

        return points_pred
        
    except Exception as e:
        print(f"Classification failed: {e}")
        return np.array([])


# Removed save_classification_npy function

# Removed save_classification_json function


def segmentation(img, model):
    IMG_WIDTH_DEST = 482
    IMG_HEIGHT_DEST = 482
    IMG_WIDTH = 128   # Reverted back to 128 to match model expectations
    IMG_HEIGHT = 128  # Reverted back to 128 to match model expectations  
    IMG_CHANNELS = 3

    if img is None:
       raise Exception("img is None")
    

    print("Segmenting image")
    original_shape = img.shape[:2]

    if original_shape != (4000, 6000):
        img = cv2.resize(img, (6000, 4000))

    reflect = cv2.copyMakeBorder(img, 184, 184, 148, 148, cv2.BORDER_REFLECT)

    pos_x = np.arange(0, 5785, 482)
    pos_y = np.arange(0, 3857, 482)
    slices = [
        np.s_[y[0] : y[1], x[0] : x[1]]
        for x in zip(pos_x, pos_x + 512)
        for y in zip(pos_y, pos_y + 512)
    ]

    print(f"Processing {len(slices)} slices in ultra-micro-batches")
    
    # Ultra-small batches
    ultra_micro_batch_size = 8
    all_preds = []
    
    for batch_start in range(0, len(slices), ultra_micro_batch_size):
        batch_end = min(batch_start + ultra_micro_batch_size, len(slices))
        batch_slices = slices[batch_start:batch_end]
        
        X = np.zeros((len(batch_slices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
        
        for j, sl in enumerate(batch_slices):
            X[j] = cv2.resize(reflect[sl], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
        
        batch_preds = model.predict(X, verbose=0, batch_size=1)
        all_preds.append(batch_preds)
    
    preds = np.vstack(all_preds)

    RESULT_Y = np.zeros(
        (len(preds), IMG_HEIGHT_DEST, IMG_WIDTH_DEST, 1), dtype=np.float32
    )

    for j, x in enumerate(preds):
        RESULT_Y[j] = np.expand_dims(
            cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)[15:497, 15:497],
            axis=-1,
        )
    
    reconstructed_prob = np.squeeze(np.hstack([np.vstack(i) for i in np.split(RESULT_Y, 13)]))[
        169:4169, 133:6133
    ]
    reconstructed_mask = (
        (reconstructed_prob > 0.5).astype(np.uint8)
        * 255
    )

    print("Resizing image")
    if original_shape != (4000, 6000):
        reconstructed_mask = cv2.resize(
            reconstructed_mask,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # remove internal areas
    print("findContours")
    # Handle both OpenCV 3.x (3 values) and 4.x (2 values) compatibility
    contours_result = cv2.findContours(reconstructed_mask, 1, 2)
    contours = contours_result[-2]  # contours is always second-to-last
    max_cnt = contours[np.argmax(np.array([cv2.contourArea(i) for i in contours]))]

    print("drawContours")
    reconstructed_mask *= 0
    cv2.drawContours(reconstructed_mask, [max_cnt], 0, (255, 255, 255), -1)

    print("boundingRect")
    bounding_rect = cv2.boundingRect(max_cnt)  # x,y,w,h

    return reconstructed_mask, bounding_rect


# Removed dir parameter from find_circles signature
def find_circles(logging, img, mask, cnt):
    try:
        x, y, w, h = cnt

        roi = np.copy(img[y : y + h, x : x + w])
        roi = cv2.split(roi)[2]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
        roi = clahe.apply(roi)
        roi = cv2.bilateralFilter(roi, 5, 50, 50)

        # find all cells with different radius
        all_points = np.array([])
        for j in range(5, 50, 5):
            points = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=2,
                minDist=12,
                param1=145,
                param2=55,
                minRadius=j + 1,
                maxRadius=j + 5,
            )

            if points is not None:
                points = points[0][:, :3].astype(np.int32)
                all_points = (
                    np.vstack((all_points, points)) if all_points.size else points
                )

        # select best radius
        if all_points.size == 0:
            best_radius = 33
        else:
            best_radius = np.bincount(all_points[:, -1]).argmax()

        minDist = best_radius * 2 - ((best_radius * 9 / 26) + 75 / 26)

        minRadius = best_radius - max(2, math.floor(best_radius * 0.1))
        maxRadius = best_radius + max(2, math.floor(best_radius * 0.1))

        # hough to find all cells
        points = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            dp=3,
            minDist=minDist,
            param1=100,
            param2=25,
            minRadius=minRadius,
            maxRadius=maxRadius,
        )

        if points is not None:
            points = points[0][:, :3]
            points[:, 2:] = points[:, 2:]
            points = points.astype(np.int32)
            points = points[points[:, 0] < w]
            points = points[points[:, 1] < h]

            points[:, 0] += x
            points[:, 1] += y

            points = points[mask[points[:, 1], points[:, 0]] > 0]

        # Removed saving npy file
        # np_name = "out.npy"
        # array_name = dir + np_name
        # np.save(array_name, points)

        # Removed saving json file
        # json_name = "out.json" #[PurePath(im_name).parts[-1].split(".")[:-1][0] + ".json"]
        # json_path = dir + json_name #os.path.join(*[PATH_DETECTIONS] + list(PurePath(im_name).parts[:-1]) + json_name)

        # with open(json_path, 'w') as f:
        #     json.dump(points.tolist(), f)
    except Exception as e:
        logging.error("Cell detection failed on image ", e) # Removed dir
    # Return the detected points
    return points


# Removed create_folder function
# def create_folder(path):
#     path = os.path.join(*PurePath(path).parts[:-1])
#     if not os.path.exists(path):
#         os.makedirs(path)

# Modified create_detections to accept image object instead of filename/dir
def create_detections(logging, img):
    logging.info("loading segmentation model...")
    model = load_segmentation_model()
    mask, cnt = segmentation(img, model)
    points = find_circles(logging, img, mask, cnt)
    return points

LABELS = ["Capped", "Eggs", "Honey", "Larves", "Nectar", "Other", "Pollen"]


def _load_max_classification_cells():
    """
    Maximum number of detected cells to classify per frame.
    Kept configurable because full-frame combs can exceed previous limits.
    """
    raw_value = os.environ.get("MAX_CLASSIFICATION_CELLS", "6000")
    try:
        parsed = int(raw_value)
        if parsed <= 0:
            raise ValueError("MAX_CLASSIFICATION_CELLS must be > 0")
        return parsed
    except Exception:
        print(
            f"Invalid MAX_CLASSIFICATION_CELLS={raw_value!r}; falling back to 6000"
        )
        return 6000


MAX_CLASSIFICATION_CELLS = _load_max_classification_cells()


# Modified classify_images to accept image object and points array
def classify_images(logging, img, points):
    logging.info("loading classification model...")
    model = load_classification_model()
    final_results = classify_image(img, points, LABELS, model, img_size)
    return final_results


# Modified run function to accept image buffer
def run(logging, image_buffer):
    try:
        logging.info("Decoding image buffer...")
        nparr = np.frombuffer(image_buffer, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Clear buffer from memory immediately
        del nparr, image_buffer
        
        if img is None:
            logging.error("Could not decode image from buffer")
            return None

        logging.info("Detecting cells...")
        points = create_detections(logging, img)
        if points is None or len(points) == 0:
            logging.info("No points detected.")
            return []

        logging.info("Classifying cells...")
        final_results = classify_images(logging, img, points)

        logging.info("Done")
        return final_results
    except Exception as e:
        logging.exception(f"Error in run function: {e}")
        return None
