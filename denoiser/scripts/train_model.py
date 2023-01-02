import argparse
import logging
import numpy as np
import tempfile
import tensorflow as tf

from denoiser.train import create_tfrecords, create_sample_from_tfrecord


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help='Input data directory. Expected to contain two subdirectories, `raw` and `clean`, containing identically' \
         'named sets of .wav files, with and without noise respectively'
)
parser.add_argument(
    '--sample-size',
    type=float,
    default=0.01,
    help='Length of each training sample, in seconds'
)
parser.add_argument(
    '--step-size',
    type=float,
    default=0.005,
    help='Step between consecutive training samples, in seconds'
)
parser.add_argument(
    '--total-samples',
    type=int,
    default=10000,
    help='The total number of samples to generate'
)
parser.add_argument(
    '--samples-per-tfrecord',
    type=int,
    default=100,
    help='Number of samples per tfrecord file'
)
parser.add_argument(
    '--noise-fraction',
    type=float,
    default=0.5,
    help='The fraction of samples that should contain noise'
)
parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='Training batch size'
)
parser.add_argument(
    '--shuffle-buffer-size',
    type=int,
    default=None,
    help='Dataset shuffle buffer size'
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

    with tempfile.TemporaryDirectory() as training_dir:
        logging.info(f"Creating training samples from {args.input_dir}")
        tfrecord_files, bitrate = create_tfrecords(
            input_dir=args.data_dir,
            output_dir=training_dir,
            sample_size=args.sample_size,
            step_size=args.step_size,
            total_samples=args.total_samples,
            samples_per_tfrecord=args.samples_per_tfrecord,
            noise_fraction=args.noise_fraction
        )

        if args.shuffle_buffer_size:
            if args.shuffle_buffer_size <= args.samples_per_tfrecord:
                logging.warning(
                    f"Data not properly shuffled: shuffle_buffer_size {args.shuffle_buffer_size} smaller than"
                    f" samples_per_tfrecord {args.samples_per_tfrecord}"
                )
            shuffle_buffer_size = args.shuffle_buffer_size
        else:
            shuffle_buffer_size = args.samples_per_tfrecord * 10  # Default to shuffling over multiple tfrecords

        np.random.shuffle(tfrecord_files)
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        dataset = dataset.map(create_sample_from_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_buffer_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(args.batch_size)