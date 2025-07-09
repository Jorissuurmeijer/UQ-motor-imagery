from moabb.paradigms import MotorImagery


def load_data(dataset, subject_id, n_classes):
    dataset = dataset
    paradigm = MotorImagery(
        n_classes=n_classes, fmin=7.5, fmax=30, tmin=0, tmax=None
    )

    subject = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subject)

    return X, y, metadata

