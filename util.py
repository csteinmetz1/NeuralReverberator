import os
import sys
import glob
import librosa
import warnings
import subprocess
from tqdm import tqdm
import soundfile as sf
import numpy as np
from sklearn.utils import resample

def load_data(dataset_dir, sequence_len, train_split=0.80, n_samples=10):
    """ 
    Utility function to load Room Impulse Responses.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset
    sequence_len : int
        Length of the RIRs when loading
    train_split : float, optional
        Fraction of the data to return as training samples

    Returns
    -------
    x_train : ndarray
        Training examples with shape (examples, audio samples)
    x_test : ndarray
        Testing examples with shape (examples, audio samples)		
    """
    IRs = [] # list to hold audio data
    load_samples = 0
    for idx, sample in enumerate(glob.glob("data_16k/*.wav")):
        if load_samples < n_samples:
            data, rate = sf.read(sample, stop=sequence_len, always_2d=True)
            data = librosa.util.fix_length(data, sequence_len, axis=0)

            for ch in range(data.shape[1]):
                IRs.append(data[:,ch])
                load_samples += 1

            sys.stdout.write("* Loaded {} RIRs\r".format(idx+1))
            sys.stdout.flush()

    x = np.stack(IRs, axis=0)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    train_idx = np.floor(n_samples*train_split).astype('int')
    x_train = x[:train_idx,:]
    x_test = x[train_idx:,:]

    print("Loaded data with shape:")
    print("x_train: {}".format(x_train.shape))
    print("x_test : {}".format(x_test.shape))

    return x_train, x_test 

def convert_sample_rate(dataset_dir, output_dir, out_sample_rate):
    """ 
    Utility function convert the sample rate of audio files.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset
    output_dir : str
        Directory to store outputs
    out_sample_rate : int
        Desired output sample rate
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    rir_list = glob.glob(os.path.join(dataset_dir, "*.wav"))
    for idx, sample in enumerate(tqdm(iterable=rir_list, desc="Converting sample rate", ncols=100)):
        filename = os.path.basename(sample).split('.')[0]
        out_filepath = os.path.join(output_dir, "{0}_{1}.wav".format(filename, out_sample_rate))
        subprocess.call("""sox "{0}" -r {1} "{2}" """.format(sample, out_sample_rate, out_filepath), shell=True, stderr=subprocess.DEVNULL)
        #sf.write(os.path.join(output_dir, filename), audio, out_sample_rate, subtype='PCM_16')

def analysis_dataset(dataset_dir):
    for idx, sample in enumerate(glob.glob("data/*.wav")):
        filename = os.path.basename(sample)
        audio, rate = sf.read(sample)
        
#convert_sample_rate('data/', 'data_16k/', 16000)