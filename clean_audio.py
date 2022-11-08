import argparse
import logging
import os
import tensorflow as tf

from datetime import datetime
from scipy.io import wavfile

from denoiser.predict import apply_rolling_average, interpolate_sequence_using_labels, make_predictions, \
    CUBIC_SPLINE, INTERPOLATION_FUNCTIONS


def _get_time_since(start: datetime) -> float:
    """ Get the time in seconds since start """
    return (datetime.now() - start).microseconds / 1000000


def _get_fit_func(fit_func: str):
    """ Get the interpolation function """
    return INTERPOLATION_FUNCTIONS.get(fit_func)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    type=str,
    required=True,
    help='Audio file to clean'
)
parser.add_argument(
    '--output',
    type=str,
    required=True,
    help='Destination for cleaned file'
)
parser.add_argument(
    '--model',
    type=str,
    required=True,
    help='Keras model (directory) to load'
)
parser.add_argument(
    '--fit-func',
    type=_get_fit_func,
    default=_get_fit_func(CUBIC_SPLINE),
    choices=INTERPOLATION_FUNCTIONS,
    help='scipy.interpolate function to use'
)
parser.add_argument(
    '--avg-window',
    type=int,
    default=10,
    help='Window size to average predictions over'
)
parser.add_argument(
    '--fit-window',
    type=int,
    default=100,
    help='Window size to fit the interpolation function over'
)
parser.add_argument(
    '--threshold',
    type=float,
    default=0.5,
    help='Probability threshold for a positive identification'
)
parser.add_argument(
    '--log',
    action='store_true'
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.log:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())

    logging.info(f"Reading file {args.input}")
    bitrate, audio_sequence = wavfile.read(os.path.join(args.input))

    logging.info(f"Loading model {args.model}")
    model = tf.keras.models.load_model(args.model)

    predict_start = datetime.now()
    preds = make_predictions(audio_sequence, model)
    logging.info("Finished predicting in {:.2f}s".format(_get_time_since(predict_start)))

    preds = apply_rolling_average(preds, args.avg_window)
    labels = (preds > args.threshold)

    interp_start = datetime.now()
    clean_sequence = interpolate_sequence_using_labels(
        audio_sequence, labels, args.fit_func, args.fit_window
    )
    logging.info("Finished interpolating in {:.2f}s".format(_get_time_since(interp_start)))

    wavfile.write(args.output, bitrate, clean_sequence)
    logging.info(f"Written to {args.output}")
