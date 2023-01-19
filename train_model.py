import argparse
import logging
import numpy as np
import tempfile

from denoiser.models import create_model
from denoiser.train import create_tfrecords, create_dataset_from_file_list


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    type=str,
    required=True,
    help='Input data directory. Expected to contain two subdirectories, `raw` and `clean`, containing identically' \
         'named sets of .wav files, with and without noise respectively'
)
parser.add_argument(
    '--model-dir',
    type=str,
    required=True,
    help='Output directory for the saved model. Can be used by clean_audio_file once trained'
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
    '--epochs',
    type=int,
    default=10,
    help='Number of training epochs'
)
parser.add_argument(
    '--validation-split',
    type=float,
    default=0.1,
    help='Fraction of samples to keep for evaluation'
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
        logging.info(f"Creating training samples from {args.data_dir}")
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
        num_val_files = int(args.validation_split*len(tfrecord_files))
        if num_val_files > 0:
            dataset = create_dataset_from_file_list(
                tfrecord_files[:-num_val_files], shuffle_buffer_size=shuffle_buffer_size, batch_size=args.batch_size
            )
            validation_dataset = create_dataset_from_file_list(
                tfrecord_files[-num_val_files:], shuffle_buffer_size=shuffle_buffer_size, batch_size=args.batch_size
            )
        else:
            dataset = create_dataset_from_file_list(
                tfrecord_files, shuffle_buffer_size=shuffle_buffer_size, batch_size=args.batch_size
            )
            validation_dataset=None

        model = create_model(sample_length=int(args.sample_size*bitrate))

        model.fit(dataset, validation_data=validation_dataset, epochs=args.epochs, verbose=int(args.log))

        model.save(args.model_dir)
        logging.info(f"Saved model to {args.model_dir}")
