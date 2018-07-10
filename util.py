import os
import sys
import glob
import librosa
import warnings
import subprocess
import keras
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.stats import norm
from sklearn.utils import resample

def load_spectrograms(dataset_dir, train_split=0.80, n_samples=3000):

    x = [] # list to hold spectrograms
    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.txt"))):
        if idx < n_samples:
            s = np.loadtxt(sample)
            if s.shape == (513, 256):
                x.append(s)
                sys.stdout.write("* Loaded {} RIR spectrograms\r".format(idx+1))
                sys.stdout.flush()

    x = np.stack(x, axis=0)
    print(x.shape)

    train_idx = np.floor(n_samples*train_split).astype('int')
    x_train = x[:train_idx,:,:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x[train_idx:,:,:]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    print("Loaded data with shape:")
    print("x_train: {}".format(x_train.shape))
    print("x_test:  {}".format(x_test.shape))

    return x_train, x_test 


def load_data(dataset_dir, sequence_len, split=True, train_split=0.80, n_samples=3000):
    """ 
    Utility function to load room impulse responses.

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
    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.wav"))):
        data, rate = sf.read(sample, stop=sequence_len, always_2d=True)
        data = librosa.util.fix_length(data, sequence_len, axis=0)

        for ch in range(data.shape[1]):
            if load_samples < n_samples:
                IRs.append(data[:,ch])
                load_samples += 1

        sys.stdout.write("* Loaded {} RIRs\r".format(idx+1))
        sys.stdout.flush()

    x = np.stack(IRs, axis=0)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

    if not split:
        return x

    train_idx = np.floor(n_samples*train_split).astype('int')
    x_train = x[:train_idx,:]
    x_test = x[train_idx:,:]

    print("Loaded data with shape:")
    print("x_train: {}".format(x_train.shape))
    print("x_test:  {}".format(x_test.shape))

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

class GenerateIRs(keras.callbacks.Callback):

    def __init__(self, n, rate, batch_size, latent_dim, decoder):
        self.n = n
        self.rate = rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.decoder = decoder

        if not os.path.isdir("results"):
            os.makedirs("results")

    def on_epoch_end(self, epoch, logs):
        epoch_dir = os.path.join("results", "epoch{}".format(epoch+1))
        if not os.path.isdir(epoch_dir):
            os.makedirs(epoch_dir)
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, self.n))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array(([xi, yi]))
                z_sample = np.tile(z_sample, self.batch_size).reshape(self.batch_size, self.latent_dim)
                x_decoded = self.decoder.predict(z_sample, batch_size=self.batch_size)
                data = np.reshape(x_decoded[0], x_decoded[0].shape[0])
                output_path = os.path.join(epoch_dir, "epoch{0}_x{1}_y{2}.wav".format(epoch+1, i, j))
                sf.write(output_path, data, self.rate)

def generate_spectrograms(dataset_dir, n_fft=1024, n_hop=256, augment_data=False):

    if not os.path.isdir("spectrograms"):
        os.makedirs("spectrograms")
   
    x = load_data(dataset_dir, 66304, split=False)

    specs_generated = 0

    for idx in range(x.shape[0]):
        s = np.reshape(x[idx,:,:], (66304,))

        if augment_data:
            data = np.reshape(x[idx,:,:], (x[idx,:,:].shape[0],))
            augmented_audio = augment_audio(data, 16000, stretch_factors=[0.80, 0.90, 1.10, 1.20], shift_factors=[-2, -1, 1, 2])
            for augment in augmented_audio:
                aug = librosa.stft(augment, n_fft=n_fft, hop_length=n_hop, center=False)
                aug = librosa.amplitude_to_db(s, ref=2.0)
                np.savetxt('spectrograms/ir_{}.txt'.format(specs_generated+1), aug)
                specs_generated += 1
        
        s = librosa.stft(s, n_fft=n_fft, hop_length=n_hop, center=False)
        s = librosa.amplitude_to_db(s, ref=2.0)
        np.savetxt('spectrograms/ir_{}.txt'.format(specs_generated), s)
        specs_generated += 1
        print("\n* Computed {} RIR spectrograms".format(specs_generated))

def augment_audio(data, rate, stretch_factors=[], shift_factors=[]):

    augmented_audio = []

    # stretch audio
    for stretch_factor in stretch_factors:
        sys.stdout.write("* Strecthing audio by {}...\r".format(stretch_factor))
        sys.stdout.flush()
        augmented_audio.append(librosa.effects.time_stretch(data, stretch_factor))
    
    for shift_factor in shift_factors:
        sys.stdout.write("* Pitching audio by {}...\r".format(shift_factor))
        sys.stdout.flush()
        augmented_audio.append(librosa.effects.pitch_shift(data, rate, shift_factor))

    return augmented_audio

def analysis_dataset(dataset_dir):
    for idx, sample in enumerate(glob.glob("data/*.wav")):
        filename = os.path.basename(sample)
        audio, rate = sf.read(sample)

generate_spectrograms('data_16k', augment_data=True)