from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from moabb.datasets import BNCI2014_001, BNCI2014_002, Zhou2016, BNCI2014_004
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras_uncertainty.utils import entropy

from project.Utils.load_data import load_data
from project.models.shallowConvNet.Deep_ensembles.SCN_model_DE import ShallowConvNet
from project.Utils.evaluate_and_plot import plot_confusion_and_evaluate, evaluate_uncertainty, plot_calibration

from tqdm import tqdm
import numpy as np

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def main():
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Number of epochs with no improvement
        mode='min',  # Minimize validation loss
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    dataset1 = BNCI2014_002()
    dataset2 = Zhou2016()
    dataset3 = BNCI2014_004()       # the bad performing one, if done again, take in range, 3,4
    dataset4 = BNCI2014_001()  # original one

    datasets = [dataset1, dataset2, dataset3, dataset4]

    n_classes = [2, 3, 2, 4]

    # This unfortunately cannot really be done more elegantly, because the paradigm to get the data needs
    #   the number of classes, and the dataset nor the dict of get_data can get the number of classes

    channels = [15, 14, 3, 22]  # the same holds here
    samples_data = [2561, 1251, 1126, 1001]

    num_models = 10

    for dataset, num_class, chans, samples in zip(datasets, n_classes, channels, samples_data):
        num_subjects = len(dataset.subject_list)

        for subject_id in range(1, num_subjects + 1):
            dataset_id = datasets.index(dataset) + 1

            X, y, metadata = load_data(dataset, subject_id, num_class)

            unique_labels = np.unique(y)
            num_unique_labels = len(unique_labels)
            assert num_unique_labels == num_class, "The number of labels does not match the expected number of classes."

            X_reshaped = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

            X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

            label_encoder = LabelEncoder()
            y_integers = label_encoder.fit_transform(y_train)
            y_categorical = np_utils.to_categorical(y_integers, num_classes=num_unique_labels)

            predictions = np.zeros((num_models, X_test.shape[0], len(np.unique(y))))

            for model_idx in tqdm(range(num_models)):

                model = ShallowConvNet(nb_classes=num_class, Chans=chans, Samples=samples, dropoutRate=0.5)
                optimizer = Adam(learning_rate=0.001)  # standard 0.001
                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                # weights = compute_sample_weight('balanced', y=y_train)    # can be used for balanced weights
                model.fit(
                    X_train,
                    y_categorical,
                    callbacks=[early_stopping],
                    epochs=100, batch_size=64, validation_split=0.1,   # sample_weighTt=weights,
                    verbose=0,
                )

                predictions[model_idx] = model.predict(X_test)

            mean_predictions = np.mean(np.array([predictions]), axis=0)

            prediction_proba = np.max(mean_predictions, axis=0)

            label_encoder = LabelEncoder()
            test_labels = label_encoder.fit_transform(y_test)

            predicted_classes = np.argmax(prediction_proba, axis=1)
            # confidence = np.max(max_pred_0, axis=1)

            entr = entropy(test_labels, prediction_proba)       # not further used for now
            print("Entropy: ", entr)

            # plot and evaluate
            plot_confusion_and_evaluate(predicted_classes, test_labels,
                                        subject_id=subject_id, dataset_id=dataset_id, save=False)

            evaluate_uncertainty(predicted_classes, test_labels, prediction_proba,
                                 subject_id=subject_id, dataset_id=dataset_id, save=False)

            plot_calibration(predicted_classes, test_labels, prediction_proba,
                             subject_id=subject_id, dataset_id=dataset_id, save=False)





if __name__ == '__main__':
    main()