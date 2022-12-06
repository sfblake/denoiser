import logging
import numpy as np
import os
from scipy.io import wavfile
import shutil
import tensorflow as tf
from typing import Tuple

from denoiser.utils import list_wavfiles, write_tfrecord


RAW_FOLDER = 'raw'
CLEAN_FOLDER = 'clean'
TFRECORD_EXTENSTION = '.tfrec'


def create_training_samples(input_dir: str, output_dir: str, sample_size: float, step_size: float,
                            num_samples: int, noise_fraction: float = 0.5) -> None:
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
    raw_files = list_wavfiles(raw_dir)
    clean_files = list_wavfiles(clean_dir)
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
        sample_ids = _get_sample_ids_from_labels(
            labels, sample_size_int, step_size_int, num_noise_samples, num_clean_samples
        )

        outfile_path = os.path.join(output_dir, os.path.splitext(file)[0]) + TFRECORD_EXTENSTION
        logging.info("Writing {} samples to {}".format(sample_ids.shape[0], outfile_path))
        with tf.io.TFRecordWriter(outfile_path) as writer:
            for sample_id in sample_ids:
                example = write_tfrecord(
                    raw_data[sample_id:sample_id + sample_size_int],
                    labels[sample_id:sample_id + sample_size_int].astype(int),
                    file=file, start_time=float(sample_id / raw_bitrate), duration=float(sample_size_int / raw_bitrate)
                )
                writer.write(example.SerializeToString())


def _get_sample_ids_from_labels(labels: np.array, sample_size: int, step_size: int,
                                num_true_samples: int, num_false_samples: int) -> np.array:
    """
    Given a 1D array of sequential boolean labels (where True is much rarer than False), generate a certain number of
    random "true" (containing a True label) and "false" (not containing a True label) samples

    Parameters
    ----------
    labels : np.array
        1D array of sequential labels
    sample_size : int
        Length of the samples to be generated
    step_size : int
        Step size between consecutive samples
    num_true_samples : int
        Number of true samples to be generated
    num_false_samples : int
        Number of false samples to be generated

    Returns
    -------
    sample_ids : np.array
        Starting sample ids
    """
    np.random.seed(0)  # Sample repeatably
    true_sample_ids, false_sample_ids = _get_indices_from_labels(labels, sample_size, step_size)
    # Filter to full samples only
    true_sample_ids = true_sample_ids[true_sample_ids + sample_size < labels.shape[0]]
    false_sample_ids = false_sample_ids[false_sample_ids + sample_size < labels.shape[0]]
    sample_ids = np.concatenate([
        np.random.choice(
            true_sample_ids, num_true_samples, replace=true_sample_ids.shape[0] < num_true_samples
        ),
        np.random.choice(
            false_sample_ids, num_false_samples, replace=false_sample_ids.shape[0] < num_false_samples
        )
    ])
    np.random.shuffle(sample_ids)
    return sample_ids


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
