import numpy as np


def read_features_from_csv(filename, usecols=range(1, 785)):
    """
    Read feature.
    :param filename: the fil name.
    :param usecols: the columns.
    :return: return the features.
    """

    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
    features = np.divide(features, 255.0)  # scale 0..255 to 0..1

    return features


def read_labels_from_csv(filename):
    """
    Read labels and convert them to 1-hot vectors.
    :param filename: the file name.
    :return: return the labels form the filename.
    """

    labels_orig = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)
    labels = np.zeros([len(labels_orig), 10])
    labels[np.arange(len(labels_orig)), labels_orig] = 1
    labels = labels.astype(np.float32)

    return labels


def generate_batch(features, labels, batch_size):
    """
    Generate Batches.
    :param features: the features data.
    :param labels: the labels data.
    :param batch_size: the batch size.
    :return: return labels and features.
    """

    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]

    return batch_features, batch_labels


def save_predicted_labels(filename, predicted_labels):
    """
    Save the predicted labels in a csv file.
    :param filename: the filename where will be save the labels.
    :param predicted_labels: the prediction labels.
    """

    predicted_labels = [np.arange(1, 1 + len(predicted_labels)), predicted_labels]
    predicted_labels = np.transpose(predicted_labels)

    np.savetxt(filename, predicted_labels, fmt='%i,%i', header='ImageId,Label', comments='')
