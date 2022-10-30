import warnings
from scipy.io.wavfile import WavFileWarning

warnings.filterwarnings('ignore', category=WavFileWarning)
