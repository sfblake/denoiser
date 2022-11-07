from datetime import datetime
import logging
import numpy as np
import os
from scipy import interpolate
from scipy.io import wavfile
from scipy.signal import convolve
import tensorflow as tf
from typing import Callable, Tuple


def clean_audio(input_path: str, output_path, model: tf.keras.Model, avg_window: int = 1, threshold: float = 0.5,
                fit_window: int = 1000):
    """
    Load data and generate noise labels for a .wav file using the specified model.

    Parameters
    ----------
    input_path : str
        Location of the wav file to
    output_path : str
        Location of the wav file to output
    model : tf.keras.Model
        Model used to label noise
    avg_window : int
        Window size to average predictions over
    threshold : float
        Probability threshold for a detection of noise
    fit_window : int
        Window size to fit the interpolation function over

    Returns
    -------
    np.array
        Cleaned audio sequence
    """
    logging.info(f"Reading file {input_path}")
    bitrate, audio_sequence = wavfile.read(os.path.join(input_path))

    predict_start = datetime.now()
    preds = _make_predictions(audio_sequence, model)
    logging.info("Finished predicting in {:.2f}s".format((datetime.now() - predict_start).microseconds/1000000))

    preds = _apply_rolling_average(preds, avg_window)
    labels = (preds > threshold)

    predict_start = datetime.now()
    clean_sequence = _interpolate_sequence_using_labels(audio_sequence, labels, interpolate.CubicSpline, fit_window)
    logging.info("Finished interpolating in {:.2f}s".format((datetime.now() - predict_start).microseconds/1000000))

    wavfile.write(output_path, bitrate, clean_sequence)
    logging.info(f"Written to {output_path}")


def _make_predictions(audio_sequence: np.array, model: tf.keras.Model) -> np.array:
    """
    Predict noise probabilities for an audio sequence using the specified model.

    Parameters
    ----------
    audio_sequence : np.array, shape (num_timesteps, 2)
        Two channel audio sequence
    model : tf.keras.Model
        Model used to label noise

    Returns
    -------
    np.array, shape (num_timesteps,)
        Noise probabilities for the audio sequence
    """
    track_length = audio_sequence.shape[0]
    sample_length = model.inputs[0].shape[1]

    # Pad to get complete samples
    num_samples = int(np.ceil(track_length / sample_length))
    extra_steps = int(num_samples * sample_length - track_length)
    data_padded = np.pad(audio_sequence, pad_width=((extra_steps, 0), (0, 0)), mode='edge')

    probs = model.predict(
        data_padded.reshape(num_samples, sample_length, 2)
    )
    return probs.reshape(-1)[extra_steps:]


def _apply_rolling_average(sequence: np.array, avg_window: int = 1) -> np.array:
    """
    Apply a rolling average with given window size to a sequence.

    Parameters
    ----------
    sequence : np.array
        1d array to apply average to
    avg_window : int
        Window size to average over

    Returns
    -------
    np.array
        Averaged sequence
    """
    return convolve(sequence, [1] * avg_window, mode='same') / avg_window


def _interpolate_sequence_using_labels(sequence: np.array, labels: np.array, func: Callable,
                                       fit_window: int = 1000) -> np.array:
    """
    Interpolate a sequence for values using a set of labels.

    Parameters
    ----------
    sequence : np.array, shape (num_timesteps, num_features)
        Sequence to interpolate
    labels : np.array, shape (num_timesteps,)
        Boolean labels for sequence, sequence will be interpolated where True
    func : callable
        Interpolation function to use, from scipy.interpolate
    fit_window : int
        Window size to fit the interpolation function over

    Returns
    -------
    np.array, shape (num_timesteps, num_features)
        Interpolated sequence
    """
    sequence = sequence.copy()
    mask = ~labels
    start_indices, end_indices = _get_periods_from_labels(labels)

    for start, end in zip(start_indices, end_indices):
        window_start = max(start - fit_window, 0)
        window_end = min(end + fit_window, len(sequence))
        fit_sequence = sequence[window_start:window_end, :]
        fit_mask = mask[window_start:window_end]
        interpolated = _masked_interpolation(fit_sequence, fit_mask, func)
        values = interpolated[start - window_start:end - window_start, :]
        sequence[start:end, :] = values

    return sequence


def _get_periods_from_labels(labels: np.array) -> Tuple[np.array, np.array]:
    """
    For a boolean array get the start and end index of periods of consecutive True values

    Parameters
    ----------
    labels : np.array
        boolean array

    Returns
    -------
    (np.array, np.array)
        Start and end indices of True periods
    """
    indices = np.argwhere(labels[1:] != labels[:-1]).reshape(-1)
    if labels[0]:
        indices = np.concatenate([[0], indices])
    if labels[-1]:
        indices = np.concatenate([indices, [labels.shape[0]]])

    return indices[::2], indices[1::2]


def _masked_interpolation(sequence: np.array, mask: np.array, func: Callable) -> np.array:
    """
    Interpolate the masked values of a sequence using the given function.

    Parameters
    ----------
    sequence : np.array, shape (num_timesteps, num_features)
        Sequence to interpolate
    mask : np.array, shape (num_timesteps,)
        Boolean sequence mask, False values will be interpolated
    func : callable
        Interpolation function to use, from scipy.interpolate

    Returns
    -------
    np.array, shape (num_timesteps, num_features)
        Interpolated sequence
    """
    sequence = sequence.copy()
    xrange = np.arange(sequence.shape[0])
    fitted_func = func(xrange[mask], sequence[mask])
    sequence[mask] = fitted_func(xrange[mask])
    return sequence
