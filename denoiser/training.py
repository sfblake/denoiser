import logging
import numpy as np
import os
from scipy.io import wavfile
import shutil
import tensorflow as tf
from typing import Tuple, Union


RAW_FOLDER = 'raw'
CLEAN_FOLDER = 'clean'
TFRECORD_EXTENSTION = '.tfrec'
DATA_FEATURE_0 = 'left'
DATA_FEATURE_1 = 'right'
LABEL = 'label'


def create_training_samples(
    input_dir: str, output_dir: str, sample_size: float, step_size: float, num_samples: int, noise_fraction: float = 0.5
) -> None:
    """
    From a set of raw (with noise) and clean (noise removed) wav files, create tfrecord files with samples for model
    training. One .tfrec file is generated per raw file.

    Parameters
    ----------
    input_dir : str
        Input data directory. Expected to contain two subdirectories, `raw` and `clean`, containing identically named
        sets of .wav files, with and without noise respectively.
    output_dir : str
        Output data directory for tfrecords.
    sample_size : float
        Length of each training sample, in seconds.
    step_size : float
        Step between consecutive training samples, in seconds.
    num_samples : int
        The number of samples to generate from each wav file.
    noise_fraction : float
        The fraction of samples that contain noise.
    """
    raw_dir = os.path.join(input_dir, RAW_FOLDER)
    clean_dir = os.path.join(input_dir, CLEAN_FOLDER)
    raw_files = os.listdir(raw_dir)
    clean_files = os.listdir(clean_dir)
    training_files = set(raw_files).intersection(set(clean_files))
    if len(training_files) == 0:
        raise FileNotFoundError("No training files found")

    if os.path.exists(output_dir):
        logging.info("Removing existing {}".format(output_dir))
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    num_noise_samples = int(num_samples * noise_fraction)
    num_clean_samples = int(num_samples * (1 - noise_fraction))
    for file in training_files:
        logging.info("Reading file {}".format(file))
        raw_bitrate, raw_data = wavfile.read(os.path.join(raw_dir, file))
        clean_bitrate, clean_data = wavfile.read(os.path.join(clean_dir, file))
        if raw_bitrate != clean_bitrate:
            raise ValueError("Bitrate does not match for file {}, {} vs {}"
                             .format(file, raw_bitrate, clean_bitrate))
        if raw_data.shape != clean_data.shape:
            raise ValueError("Duration/channels do not match for file {}, {} vs {}"
                             .format(file, raw_data.shape, clean_data.shape))

        sample_size_int = int(sample_size * raw_bitrate)
        step_size_int = int(step_size * raw_bitrate)
        labels = np.not_equal(raw_data, clean_data).any(axis=1)  # Timestep is noise if raw data does not match clean
        noise_sample_ids, clean_sample_ids = _get_indices_from_labels(labels, sample_size_int, step_size_int)
        np.random.seed(0)  # Sample repeatably
        sample_ids = np.concatenate([
            np.random.choice(
                noise_sample_ids, num_noise_samples, replace=noise_sample_ids.shape[0] < num_noise_samples
            ),
            np.random.choice(
                clean_sample_ids, num_clean_samples, replace=clean_sample_ids.shape[0] < num_clean_samples
            )
        ])

        outfile_path = os.path.join(output_dir, os.path.splitext(file)[0]) + TFRECORD_EXTENSTION
        logging.info("Writing {} samples to {}".format(sample_ids.shape[0], outfile_path))
        with tf.io.TFRecordWriter(outfile_path) as writer:
            for sample_id in sample_ids:
                example = _create_tfrecord(
                    raw_data[sample_id:sample_id + sample_size_int],
                    labels[sample_id:sample_id + sample_size_int].astype(int),
                    start_time=float(sample_id / raw_bitrate),
                    duration=float(sample_size_int / raw_bitrate),
                    file=file
                )
                writer.write(example.SerializeToString())


def _get_indices_from_labels(labels: np.array, sample_size: int, step_size: int) -> Tuple[np.array, np.array]:
    """
    Given a 1D array of sequential boolean labels (where True is much rarer than False),
    find the starting indices of samples which will contain a True label

    Parameters
    ----------
    labels : np.array
        1D array of sequential labels
    sample_size : int
        Length of the samples to be generated
    step_size : int
        Step size between consecutive samples

    Returns
    -------
    true_sample_ids : np.array
        Starting sample ids containing a true label
    false_sample_ids : np.array
        Starting sample ids not containing a true label
    """
    sample_ids = np.arange(0, labels.shape[0], step_size)  # Possible starting sample ids
    true_label_ids = np.argwhere(labels)
    dist_to_true = true_label_ids - sample_ids  # Distance from each sample start to each true label
    next_true_label = np.where(dist_to_true >= 0, dist_to_true, np.inf).min(axis=0)  # Next true label for each sample
    true_sample_ids = sample_ids[next_true_label < sample_size]
    false_sample_ids = sample_ids[next_true_label >= sample_size]
    return true_sample_ids, false_sample_ids


def _create_tfrecord(data: np.array, labels: np.array, **kwargs) -> tf.train.Example:
    """
    Create a tfrecord from two channel audio data and a set of corresponding labels.

    Parameters
    ----------
    data : np.array
        2 channel audio data, shape (num_timesteps, 2)
    labels : np.array
        Corresponding timestep laebls, shape (num_timesteps,)
    **kwargs
        Any other single-valued variables to be added to the tfrecord

    Returns
    -------
    tf.train.Example
        tf example for serialising to tf record
    """

    feature = {
        k: _get_single_tf_feature(v)
        for k, v in kwargs.items()
    }
    feature[DATA_FEATURE_0] = tf.train.Feature(float_list=tf.train.FloatList(value=data[:, 0]))
    feature[DATA_FEATURE_1] = tf.train.Feature(float_list=tf.train.FloatList(value=data[:, 1]))
    feature[LABEL] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _get_single_tf_feature(x: Union[str, float, int]) -> tf.train.Feature:
    """ Create a tensorflow feature from a single value """
    if type(x) is str:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[x.encode('utf-8')])
        )
    elif type(x) is float:
        return tf.train.Feature(
            float_list=tf.train.FloatList(value=[x])
        )
    elif type(x) is int:
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[x])
        )
    else:
        raise TypeError("Unknown tf feature type for {}".format(type(x)))
