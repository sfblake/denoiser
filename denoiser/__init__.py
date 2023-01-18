import absl.logging
import os
import warnings
from scipy.io.wavfile import WavFileWarning

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=WavFileWarning)