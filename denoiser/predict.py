from datetime import datetime
import logging
import numpy as np
import os
from scipy.io import wavfile
import tensorflow as tf


def process_file(path_to_file: str, model: tf.keras.Model, model_bitrate: int):
    """
    -

    Parameters
    ----------
    path_to_file : str
        Location of the wav file to process
    model : tf.keras.Model
        Model used to label noise
    model_bitrate : int
        Bitrate of the model used

    Returns
    -------
    preds : np.array
        Noise labels for the file
    """
    logging.info("Reading file {}".format(path_to_file))
    bitrate, data = wavfile.read(os.path.join(path_to_file))
    if bitrate != model_bitrate:
        raise ValueError("File bitrate {} does not match model bitrate {}"
                         .format(bitrate, model_bitrate))
    track_length = data.shape[0]
    sample_length = model.inputs[0].shape[1]

    # Pad to get complete samples
    num_samples = int(np.ceil(track_length / sample_length))
    extra_steps = int(num_samples * sample_length - track_length)
    front_pad = np.floor(extra_steps / 2).astype(int)
    back_pad = np.ceil(extra_steps / 2).astype(int)
    data_padded = np.pad(data, pad_width=((front_pad, back_pad), (0, 0)), mode='edge')
    logging.info("File length {:.0f}s split into {} samples"
                 .format(track_length/bitrate, num_samples))

    predict_start = datetime.now()
    preds = model.predict(
        data_padded.reshape(num_samples, sample_length, 2)
    )
    preds = preds.reshape(-1)[front_pad:-back_pad]
    logging.info("Finished predicting in {}s"
                 .format((datetime.now() - predict_start).seconds))
    return preds
