import logging
import numpy as np
import os
from scipy.io import wavfile
import shutil
import tensorflow as tf
from typing import Tuple

from denoiser.utils import list_wavfiles, read_tfrecord, write_tfrecord, DATA_KEY_0, DATA_KEY_1, LABEL_KEY


RAW_FOLDER = 'raw'
CLEAN_FOLDER = 'clean'
TFRECORD_EXTENSTION = '.tfrec'


def create_tfrecords(input_dir: str, output_dir: str, sample_size: float, step_size: float, num_samples: int,
                     noise_fraction: float = 0.5) -> Tuple[list, int]:
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
        The fraction of samples that should contain noise.

    Returns
    -------
    output_files : list
        List of tfrecord output files
    bitrate : int
        Bit rate of the audio data
    """
    raw_files = list_wavfiles(os.path.join(input_dir, RAW_FOLDER))
    clean_files = list_wavfiles(os.path.join(input_dir, CLEAN_FOLDER))
    training_files = set(raw_files).intersection(set(clean_files))
    if len(training_files) == 0:
        raise FileNotFoundError("No training files found")

    if os.path.exists(output_dir):
        logging.info("Removing existing {}".format(output_dir))
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    num_noise_samples = int(num_samples * noise_fraction)
    num_clean_samples = int(num_samples * (1 - noise_fraction))
    output_files = []
    bitrate = None
    for file in training_files:
        raw_data, clean_data, file_bitrate = _read_files_and_check_bitrate(file, input_dir)
        if bitrate and (file_bitrate != bitrate):
            raise ValueError(
                f"Bitrate for file {file} does not match previous files, {file_bitrate} vs {bitrate}"
            )
        bitrate = file_bitrate
        
        sample_size_int = int(sample_size * bitrate)
        step_size_int = int(step_size * bitrate)
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
                    file=file, start_time=float(sample_id / bitrate), duration=float(sample_size_int / bitrate)
                )
                writer.write(example.SerializeToString())
        output_files.append(outfile_path)

    return output_files, bitrate


def create_sample_from_tfrecord(example: tf.train.Example) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convert a tfrecord example into a training sample and label.

    Parameters
    ----------
    example: tf.train.Example
        tf example read from tf record

    Returns
    -------
    tf.Tensor
        2 channel audio data, shape (num_timesteps, 2)
    tf.Tensor
        Corresponding timestep labels, shape (num_timesteps,)
    """
    example = read_tfrecord(example)
    return tf.stack([example[DATA_KEY_0], example[DATA_KEY_1]], axis=-1), example[LABEL_KEY]


def _get_sample_ids_from_labels(labels: np.Array, sample_size: int, step_size: int,
                                num_true_samples: int, num_false_samples: int) -> np.Array:
    """
    Given a 1D array of sequential boolean labels (where True is much rarer than False), generate a certain number of
    random "true" (containing a True label) and "false" (not containing a True label) samples

    Parameters
    ----------
    labels : np.Array
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
    sample_ids : np.Array
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


def _get_indices_from_labels(labels: np.Array, sample_size: int, step_size: int) -> Tuple[np.Array, np.Array]:
    """
    Given a 1D array of sequential boolean labels (where True is much rarer than False),
    find the starting indices of samples which will contain a True label

    Parameters
    ----------
    labels : np.Array
        1D array of sequential labels
    sample_size : int
        Length of the samples to be generated
    step_size : int
        Step size between consecutive samples

    Returns
    -------
    true_sample_ids : np.Array
        Starting sample ids containing a true label
    false_sample_ids : np.Array
        Starting sample ids not containing a true label
    """
    sample_ids = np.arange(0, labels.shape[0], step_size)  # Possible starting sample ids
    true_label_ids = np.argwhere(labels)
    dist_to_true = true_label_ids - sample_ids  # Distance from each sample start to each true label
    next_true_label = np.where(dist_to_true >= 0, dist_to_true, np.inf).min(axis=0)  # Next true label for each sample
    true_sample_ids = sample_ids[next_true_label < sample_size]
    false_sample_ids = sample_ids[next_true_label >= sample_size]
    return true_sample_ids, false_sample_ids


def _read_files_and_check_bitrate(file: str, input_dir: str) -> Tuple[np.Array, np.Array, int]:
    """
    Read raw and clean versions of the same wavfile, and check their length and bit rate match.

    Parameters
    ----------
    file : str
        Filename to read.
    input_dir : str
        Input data directory. Expected to contain two subdirectories, `raw` and `clean`, containing identically named
        sets of .wav files, with and without noise respectively.

    Returns
    -------
    raw_data : np.Array
        Audio data from the raw file
    clean_data : np.Array
        Audio data from the clean file
    bitrate : int
        Bit rate of the audio data
    """
    logging.info("Reading file {}".format(file))
    raw_bitrate, raw_data = wavfile.read(
        os.path.join(os.path.join(input_dir, RAW_FOLDER), file)
    )
    clean_bitrate, clean_data = wavfile.read(
        os.path.join(os.path.join(input_dir, CLEAN_FOLDER), file)
    )
    if raw_bitrate != clean_bitrate:
        raise ValueError(
            f"Bitrate does not match for file {file}, {raw_bitrate} vs {clean_bitrate}"
        )
    if raw_data.shape != clean_data.shape:
        raise ValueError(
            f"Duration/channels do not match for file {file}, {raw_data.shape} vs {clean_data.shape}"
        )
    
    return raw_data, clean_data, raw_bitrate
