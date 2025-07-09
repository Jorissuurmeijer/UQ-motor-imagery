from itertools import tee
import numpy as np
from keras_uncertainty.utils import classifier_calibration_error, classifier_calibration_curve
from matplotlib import pyplot as plt

'''
This file is based on the calibration.py file in the keras_uncertainty library 
made by Matias Valdenegro-toro, which can be found here:
https://github.com/mvaldenegro/keras-uncertainty/blob/master/keras_uncertainty/utils/calibration.py
'''

EPSILON = 1e-5


# From itertools recipes
def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def accuracy(y_true, y_pred):
    """
        Simple categorical accuracy.
    """
    return np.mean(y_true == y_pred)


def get_calibration_curve(y_pred, y_true, y_confidences, metric="mae", num_bins=10):
    """
        Estimates the calibration plot for a classifier and returns the points in the plot.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """
    return classifier_calibration_curve(y_pred, y_true, y_confidences, metric, num_bins)


def plot_calibration_curve(y_pred, y_true, y_confidences, subject_id, dataset_id, metric="mae", num_bins=10, save=True):
    """
        Plots the calibration curve for a classifier.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """
    x, y = get_calibration_curve(y_pred, y_true, y_confidences, metric, num_bins)

    plt.plot(x, y, color='red', alpha=1, linewidth=2)
    plt.plot([0, 1], [0, 1], color='black', alpha=0.2)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Plot subject {subject_id}, dataset {dataset_id}")
    if save:
        plt.savefig(f"./graphs/calibration_plots/dataset{dataset_id}/calibration_subject{subject_id}.png")
    else:
        plt.show()
    plt.clf()

    return x, y


def get_ece(y_pred, y_true, y_confidences, metric="mae", num_bins=10, weighted=False):
    """
        Estimates calibration error for a classifier.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """
    return classifier_calibration_error(y_pred, y_true, y_confidences, metric, num_bins, weighted)


def get_mce(y_pred, y_true, y_confidences, num_bins=10, weighted=False):
    """
        Estimates Maximum calibration error for a classifier.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)

    errors = []
    weights = []

    for start, end in pairwise(bin_edges):
        indices = np.where(np.logical_and(y_confidences >= start, y_confidences < end))
        filt_preds = y_pred[indices]
        filt_classes = y_true[indices]
        filt_confs = y_confidences[indices]

        if len(filt_confs) > 0:
            bin_acc = accuracy(filt_classes, filt_preds)
            bin_conf = np.mean(filt_confs)

            error = abs(bin_conf - bin_acc)
            weight = len(filt_confs)

            errors.append(error)
            weights.append(weight)

    errors = np.array(errors)
    weights = np.array(weights) / sum(weights)

    if weighted:
        return sum(errors * weights)

    return np.max(errors)


def get_nce(y_pred, y_true, y_confidences, num_bins=10, weighted=False):
    """
        Estimates Net calibration error for a classifier.
        The definition for Net calibration error can be found in the paper by Groot, T. (2024).
        Confidence is Key: utils Estimation in Large Language Models and Vision Language Models (Doctoral dissertation).

        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)

    errors = []
    weights = []

    for start, end in pairwise(bin_edges):
        indices = np.where(np.logical_and(y_confidences >= start, y_confidences < end))
        filt_preds = y_pred[indices]
        filt_classes = y_true[indices]
        filt_confs = y_confidences[indices]

        if len(filt_confs) > 0:
            bin_acc = accuracy(filt_classes, filt_preds)
            bin_conf = np.mean(filt_confs)

            error = (bin_acc - bin_conf)   # no abs because NCE does take direction
            weight = len(filt_confs)

            errors.append(error)
            weights.append(weight)

    errors = np.array(errors)
    weights = np.array(weights) / sum(weights)

    if weighted:
        return sum(errors * weights)

    return np.mean(errors)
