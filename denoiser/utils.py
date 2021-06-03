import numpy as np
import tensorflow as tf
from typing import List


DATA_KEY_0 = 'left_channel'
DATA_KEY_1 = 'right_channel'
LABEL_KEY = 'label'
FILE_KEY = 'filename'
START_KEY = 'start_time_sec'
LEN_KEY = 'duration_sec'


def write_tfrecord(data: np.array, labels: np.array, file: str, start_time: float, duration: float) -> tf.train.Example:
    """
    Create a tfrecord from two channel audio data and a set of corresponding labels.

    Parameters
    ----------
    data : np.array
        2 channel audio data, shape (num_timesteps, 2)
    labels : np.array
        Corresponding timestep laebls, shape (num_timesteps,)
    file : str
        Name of wav file the sample is taken from
    start_time : float
        Start time of the sample in the wav file, in seconds
    duration : float
        Duration of the sample, in seconds

    Returns
    -------
    tf.train.Example
        tf example for serialising to tf record
    """
    features = {
        DATA_KEY_0: _tf_float_feature(data[:, 0]),
        DATA_KEY_1: _tf_float_feature(data[:, 1]),
        LABEL_KEY: _tf_int_feature(labels),
        FILE_KEY: _tf_str_feature([file]),
        START_KEY: _tf_float_feature([start_time]),
        LEN_KEY: _tf_float_feature([duration])
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def read_tfrecord(example: tf.train.Example) -> dict:
    """
    Read a tfrecord

    Parameters
    ----------
    example: tf.train.Example
        tf example read from tf record

    Returns
    -------
    dict
        Contents of the tf record
    """
    features = {
        DATA_KEY_0: tf.io.VarLenFeature(tf.float32),
        DATA_KEY_1: tf.io.VarLenFeature(tf.float32),
        LABEL_KEY: tf.io.VarLenFeature(tf.int64),
        FILE_KEY: tf.io.FixedLenFeature([], tf.string),
        START_KEY: tf.io.FixedLenFeature([], tf.float32),
        LEN_KEY: tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example, features)
    for key in [DATA_KEY_0, DATA_KEY_1, LABEL_KEY]:
        example[key] = tf.sparse.to_dense(example[key])
    return example


def _tf_float_feature(x: List[float]) -> tf.train.Feature:
    """ Create a float tensorflow feature """
    return tf.train.Feature(float_list=tf.train.FloatList(value=x))


def _tf_int_feature(x: List[int]) -> tf.train.Feature:
    """ Create an int tensorflow feature """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=x))


def _tf_str_feature(x: List[str]) -> tf.train.Feature:
    """ Create a string tensorflow feature """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[s.encode('utf-8') for s in x]))
