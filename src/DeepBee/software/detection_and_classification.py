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

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

# from tensorflow.keras.models import model_from_json, load_model
from keras.models import load_model, model_from_json
from keras.applications.imagenet_utils import preprocess_input

import math
from collections import Counter
import datetime
import warnings
import imghdr
from pathlib import PurePath

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

PATH = os.path.dirname(os.path.realpath("__file__"))

ROOT = '/app'
# ROOT = '/Users/artjom/git/models-frame-resources/'
# ROOT = '/home/artjom/git/models-frame-resources/'

PATH_SEG_MODEL = f'{ROOT}/src/DeepBee/software/model/segmentation.h5'
PATH_SEG_MODEL_JSON = f'{ROOT}/src/DeepBee/software/model/segmentation.model.json'
PATH_SEG_MODEL_WEIGHTS = f'{ROOT}/src/DeepBee/software/model/segmentation.weights.h5'

PATH_CL_MODEL = f'{ROOT}/src/DeepBee/software/model/classification.h5'
PATH_CL_MODEL_JSON = f'{ROOT}/src/DeepBee/software/model/classification.model.json'
PATH_CL_MODEL_WEIGHTS = f'{ROOT}/src/DeepBee/software/model/classification.weights.h5'

PATH_IMAGES = f'{ROOT}/src/DeepBee/original_images/'
PATH_DETECTIONS = f'{ROOT}/src/DeepBee/annotations/detections/'
PATH_PREDICTIONS = f'{ROOT}/src/DeepBee/annotations/predictions/'
PATH_OUT_IMAGE = f'{ROOT}/src/DeepBee/output/labeled_images/'
PATH_OUT_CSV = f'{ROOT}/src/DeepBee/output/spreadsheet/'

# PATH_IMAGES = os.path.join(*list(PurePath("../original_images/").parts))
# PATH_MODEL = "/app/src/DeepBee/software/model/"
# PATH_MODEL = "model"
# PATH_DETECTIONS = os.path.join(*list(PurePath("../annotations/detections/").parts))
# PATH_PREDICTIONS = os.path.join(*list(PurePath("../annotations/predictions/").parts))
# PATH_OUT_IMAGE = os.path.join(*list(PurePath("../output/labeled_images/").parts))
# PATH_OUT_CSV = os.path.join(*list(PurePath("../output/spreadsheet/").parts))
MIN_CONFIDENCE = 0.9995

LEFT_BAR_SIZE = 480
img_size = 224
batch_size = 100

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
        size_border = pts[:, 2].max() + 1
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
                (224, 224),
            )
            for i in pts
        ]

    return ROIs


def classify_image(im_name, npy_name, labels, net, img_size, file):
    try:
        if not os.path.isfile(im_name):
            raise
        if not os.path.isfile(npy_name):
            raise

        image = cv2.imread(im_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        points = np.load(npy_name)

        pt = np.copy(points)
        pt[:, 2] = pt[:, 2] // 2

        blob_imgs = extract_circles(image, np.copy(pt), output_size=img_size)
        blob_imgs = np.asarray([i for i in blob_imgs])
        blob_imgs = preprocess_input(blob_imgs)

        scores = None

        for chunk in [
            blob_imgs[x : x + batch_size] for x in range(0, len(blob_imgs), batch_size)
        ]:
            output = net.predict(chunk)

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))

        lb_predictions = np.argmax(scores, axis=1)
        vals_predictions = np.amax(scores, axis=1)

        points_pred = np.hstack(
            (np.copy(points), np.expand_dims(lb_predictions, axis=0).T)
        )

        sum_predictions = Counter(lb_predictions)
        lb = [j + " " + str(sum_predictions[i]) for i, j in enumerate(labels)]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_predita = draw_circles_labels(image, lb, points_pred)

        inside_roi = np.ones_like(points_pred[:, 3])
        new_class = np.copy(points_pred[:, 3])

        st_use_retrain = (vals_predictions > MIN_CONFIDENCE) * 1

        csl = np.vstack(
            [i for i in [new_class, st_use_retrain, inside_roi, vals_predictions]]
        ).T

        points_pred = np.hstack((points_pred, csl))

        if file is not None:
            file.write(
                ",".join(
                    [im_name.split("/")[-1], *get_qtd_by_class(points_pred, labels)]
                )
                + "\n"
            )

        date_saved = datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")
        height, width, _ = image.shape
        roi = ((0, 0), (width, height))

        save_classification_npy(roi, date_saved, points_pred, im_name)
        save_classification_json(roi, date_saved, points_pred, im_name)

        # save as image
        out_img_name = os.path.join(PATH_OUT_IMAGE, im_name.replace(PATH_IMAGES, ""))
        create_folder(out_img_name)
        cv2.imwrite(out_img_name, cv2.resize(img_predita, (1500, 1000)))
    except Exception as e:
        print("\nFiled to classify image " + im_name, e)

def save_classification_npy(roi, date_saved, points_pred, im_name):
    # save as npy
    array_to_save = np.array([roi, date_saved, points_pred])

    if PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]:
        dest_folder = os.path.join(
            PATH_PREDICTIONS,
            os.path.join(*PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]),
        )
    else:
        dest_folder = PATH_PREDICTIONS

    array_name = PurePath(im_name).parts[-1].split(".")[:-1][0] + ".npy"
    array_name = os.path.join(dest_folder, array_name)

    create_folder(array_name)
    np.save(array_name, array_to_save)

def save_classification_json(roi, date_saved, points_pred, im_name):
    """
        x_coordinates
        y_coordinates
        radii
        predicted_labels
        new_class -  A new class label for each circle, based on a threshold confidence value.
        st_use_retrain - A binary value (0 or 1) that indicates whether the circle should be used for retraining the model, based on the confidence value.
        inside_roi - A binary value (0 or 1) that indicates whether the circle is inside the region of interest (ROI) of the input image.
        vals_predictions - The maximum score value for each circle, obtained from the neural network predictions.

    """
    # Save as JSON
    array_to_save = {
        'roi':roi, 
        'date_saved':date_saved, 
        'points_pred':points_pred
    }

    if PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]:
        dest_folder = os.path.join(
            PATH_PREDICTIONS,
            os.path.join(*PurePath(im_name.replace(PATH_IMAGES, "")).parts[:-1]),
        )
    else:
        dest_folder = PATH_PREDICTIONS

    array_name = PurePath(im_name).parts[-1].split(".")[:-1][0] + ".json"
    array_name = os.path.join(dest_folder, array_name)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    create_folder(array_name)
    with open(array_name, "w") as f:
        json.dump(array_to_save, f, 
            cls=NumpyEncoder,
            separators=(',', ':'), 
            sort_keys=True, 
            indent=4)


def segmentation(img, model):
    IMG_WIDTH_DEST = 482
    IMG_HEIGHT_DEST = 482
    IMG_WIDTH = 128
    IMG_HEIGHT = 128
    IMG_CHANNELS = 3

    print("Segmenting image")
    print(img)
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

    print("Predicting slices")
    X = np.zeros((len(slices), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

    for j, sl in enumerate(slices):
        X[j] = cv2.resize(
            reflect[sl], (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA
        )

    preds = model.predict(X)
    preds = (preds > 0.5).astype(np.uint8)

    RESULT_Y = np.zeros(
        (len(preds), IMG_HEIGHT_DEST, IMG_WIDTH_DEST, 1), dtype=np.uint8
    )

    for j, x in enumerate(preds):
        RESULT_Y[j] = np.expand_dims(
            cv2.resize(x, (512, 512), interpolation=cv2.INTER_LINEAR)[15:497, 15:497],
            axis=-1,
        )

    reconstructed_mask = (
        np.squeeze(np.hstack([np.vstack(i) for i in np.split(RESULT_Y, 13)]))[
            169:4169, 133:6133
        ]
        * 255
    )

    print("Resizing image")
    if original_shape != (4000, 6000):
        reconstructed_mask = cv2.resize(
            reconstructed_mask, (original_shape[1], original_shape[0])
        )

    # remove internal areas
    _, contours, _ = cv2.findContours(reconstructed_mask, 1, 2)
    max_cnt = contours[np.argmax(np.array([cv2.contourArea(i) for i in contours]))]

    print("drawing contours")
    reconstructed_mask *= 0
    cv2.drawContours(reconstructed_mask, [max_cnt], 0, (255, 255, 255), -1)

    bounding_rect = cv2.boundingRect(max_cnt)  # x,y,w,h

    return reconstructed_mask, bounding_rect


def find_circles(im_name, img, mask, cnt):
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

        # save as npy
        np_name = [PurePath(im_name).parts[-1].split(".")[:-1][0] + ".npy"]
        array_name = os.path.join(
            *[PATH_DETECTIONS] + list(PurePath(im_name).parts[:-1]) + np_name
        )
        create_folder(array_name)
        np.save(array_name, points)

        # save as json
        json_name = [PurePath(im_name).parts[-1].split(".")[:-1][0] + ".json"]
        json_path = os.path.join(*[PATH_DETECTIONS] + list(PurePath(im_name).parts[:-1]) + json_name)
        create_folder(json_path)

        with open(json_path, 'w') as f:
            json.dump(points.tolist(), f)
    except:
        print("Cell detection failed on image ", PurePath(im_name).parts[-1] + "\n")


def create_folder(path):
    path = os.path.join(*PurePath(path).parts[:-1])
    if not os.path.exists(path):
        os.makedirs(path)


def find_image_names():
    l_images = []
    for path, subdirs, files in os.walk(PATH_IMAGES):
        for name in files:
            full_path = os.path.join(path, name)
            if imghdr.what(full_path) is not None:
                l_images.append(full_path.replace(PATH_IMAGES, ""))
    return l_images

def create_detections():
    images = find_image_names()

    print("loading model...")
    # model = load_model(PATH_SEG_MODEL)
    with open(PATH_SEG_MODEL_JSON, 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(PATH_SEG_MODEL_WEIGHTS)

        print("creating detections...")
        print(images);
        
        for i in images:
            print("image: " + i)
            img = cv2.imread(os.path.join(PATH_IMAGES, i))
            print("image read")
            mask, cnt = segmentation(img, model)
            print("segmentation done")
            find_circles(i, img, mask, cnt)

LABELS = ["Capped", "Eggs", "Honey", "Larves", "Nectar", "Other", "Pollen"]


def classify_images():
    images = sorted([os.path.join(PATH_IMAGES, i) for i in find_image_names()])
    print(images)

    find_image_detections = lambda i: ".".join(i.split(".")[:-1]) + ".npy"

    detections = [
        os.path.join(PATH_DETECTIONS, find_image_detections(i).replace(PATH_IMAGES, ""))
        for i in images
    ]
    
    # model = load_model(PATH_CL_MODEL)
    with open(PATH_CL_MODEL_JSON, 'r') as json_file:
        model_json = json_file.read()
        model = model_from_json(model_json)
        model.load_weights(PATH_CL_MODEL_WEIGHTS)

        for i, j in zip(images, detections):
            classify_image(i, j, LABELS, model, img_size, None)


def run():
    # cross_plataform_directory()
    print("\nDetecting cells...")
    create_detections()
    print("\nClassifying cells...")
    classify_images()
    print("Done.")


if __name__ == "__main__":
    run()
