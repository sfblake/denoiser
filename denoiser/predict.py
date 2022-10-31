from datetime import datetime
import logging
import numpy as np
import os
from scipy.io import wavfile
from scipy.signal import convolve
import tensorflow as tf
from typing import Callable


def get_audio_and_labels_from_file(path_to_file: str, model: tf.keras.Model,
                                   avg_window: int = 1, threshold: float = 0.5):
    """
    Load data and generate noise labels for a .wav file using the specified model.

    Parameters
    ----------
    path_to_file : str
        Location of the wav file to process
    model : tf.keras.Model
        Model used to label noise
    avg_window : int
        Window size to average predictions over
    threshold : float
        Probability threshold for a detection of noise

    Returns
    -------
    preds : np.array
        Noise labels for the file
    """
    logging.info("Reading file {}".format(path_to_file))
    _, audio_sequence = wavfile.read(os.path.join(path_to_file))

    predict_start = datetime.now()
    preds = _make_predictions(audio_sequence, model)
    logging.info("Finished predicting in {:.1f}s".format((datetime.now() - predict_start).seconds))

    preds = convolve(preds, [1]*avg_window, mode='same') / avg_window
    return audio_sequence, (preds > threshold).astype(int)


def _make_predictions(audio_sequence: np.array, model: tf.keras.Model):
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


def _masked_interpolation(sequence: np.array, mask: np.array, func: Callable):
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

