from sklearn.utils.extmath import softmax

from project.utils.calibration import get_ece

import numpy as np


def find_best_temperature(predictions, y_true, distances):
    temperatures = np.linspace(0.0001, 10, 1000)  # search over these values
    best_ece = float('inf')
    best_temperature = 1.0
    for temp in temperatures:
        prediction_proba = softmax(distances / temp)
        confidence = np.max(prediction_proba, axis=1)
        ece = get_ece(predictions, y_true, confidence)
        if ece < best_ece:
            best_ece = ece
            best_temperature = temp

    # assert best_temperature > 0.0001
    # assert best_temperature < 10
    return best_temperature
