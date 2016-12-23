import numpy as np
import csv
import pickle
import os.path as path


def read_features_from_csv(filename, usecols=range(1, 785)):
    """
    Read feature.
    :param filename: the fil name.
    :param usecols: the columns.
    :return: return the features.
    """

    return np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)


def read_labels_from_csv(filename):
    """
    Read labels and convert them to 1-hot vectors.
    :param filename: the file name.
    :return: return the labels form the filename.
    """

    return np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int32)


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


def create_file(filename):
    """
    Create the file name.
    :param filename: the name of the file.
    :return: nothing.
    """

    with open(filename, "w") as csvfile:
        fieldnames = ['ImageId', 'Label']

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def append_data_to_file(filename, predicted_labels, start):
    """
    Append the predicted labels in a csv file.
    :param filename: the filename where will be append the labels.
    :param predicted_labels: the prediction labels.
    :param start: the start point.
     """

    predicted_labels = [np.arange(start + 1, 1 + start + len(predicted_labels)), predicted_labels]
    predicted_labels = np.transpose(predicted_labels)

    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(label for label in predicted_labels)


def write_to_file(filename, predicted_labels):
    """
    Wirete the labels in a csv file.
    :param filename: the filename where will be append the labels.
    :param predicted_labels: the prediction labels.
    """

    predicted_labels = [np.arange(1, 1 + len(predicted_labels)), predicted_labels]
    predicted_labels = np.transpose(predicted_labels)

    with open(filename, 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(label for label in predicted_labels)


def unpickle(file):
    """
    :param file: the path to the data/test batch
    :return: return a dictionary which contain data and labels from a batch
    """
    with open(file, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        dict = u.load()

        return dict


def pickle_nn(file, nn):
    """
    Save the learned data from a Neural Network
    :param file: the place where will be saved the data
    :param nn: a dictionary which contains all data
    :return: return true if the file was saved else it return false
    """

    with open(file, 'wb') as f:
        pickle.dump(nn, f)

    if path.isfile(file):
        return True
    else:
        return False


def file_exist(file):
    """
    Check if a file exist in the current computer
    :param file: the file
    :return: return tru if the file exits else return false
    """

    if path.isfile(file):
        return True
    else:
        return False
