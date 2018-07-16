import os
import sys
import csv
import glob
import librosa
import shutil
import warnings
import subprocess
import keras
import numpy as np
from tqdm import tqdm
import soundfile as sf
from scipy.stats import norm
from sklearn.utils import resample

import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt
import librosa.display

def load_spectrograms(dataset_dir, train_split=0.80, n_samples=None):
    """
    Utility function to load spectogram data.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset.
    train_split : float, optional
        Fraction of the data to return as training samples.
    n_samples : int, optional
        Number of total dataset examples to load. 
        (Deafults to full size of the dataset)

    Returns
    -------
    x_train : ndarray
        Training set (n_samples, n_freq_bins, n_time).
    x_test : ndarray
        Testing set (n_samples, n_freq_bins, n_time).
    """
    if n_samples is None: # set number of samples to full dataset
        n_samples = len(glob.glob(os.path.join(dataset_dir, "*.txt")))

    x = [] # list to hold spectrograms
    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.txt"))):
        if idx < n_samples:
            s = np.loadtxt(sample)

            if s.shape[1] < 256: # pad the input to be of shape (513, 256)
                out = np.zeros((513, 256))
                out[:s.shape[0],:s.shape[1]] = s
            else: # crop the input to be of shape (513, 256)
                out = s[:,:256]

            x.append(out)
            sys.stdout.write(f"* Loaded {idx+1}/{n_samples} RIR spectrograms\r")
            sys.stdout.flush()

    x = np.stack(x, axis=0)

    train_idx = np.floor(n_samples*train_split).astype('int')
    x_train = x[:train_idx,:,:]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_test = x[train_idx:,:,:]
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

    print("x_train: {}".format(x_train.shape))
    print("x_test:  {}".format(x_test.shape))

    return x_train, x_test 


def load_data(dataset_dir, split=True, train_split=0.80):
    """ 
    Utility function to load room impulse responses.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset.
    train_split : float, optional
        Fraction of the data to return as training samples.

    Returns
    -------
    x_train : ndarray
        Training examples with shape (examples, audio samples).
    x_test : ndarray
        Testing examples with shape (examples, audio samples).	
    """
    IRs = [] # list to hold audio data
    sample_names = [] # temp list - delete this later
    load_samples = 0
    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.wav"))):
        data, rate = sf.read(sample, always_2d=True)

        for ch in range(data.shape[1]):
            IRs.append(data[:,ch])
            load_samples += 1
            sample_names.append(os.path.basename(sample).replace('.wav', ''))

        sys.stdout.write("* Loaded {} RIRs\r".format(load_samples+1))
        sys.stdout.flush()

    if not split:
        return IRs, sample_names

    x = np.stack(IRs, axis=0)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))

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
        Directory containing the dataset.
    output_dir : str
        Directory to store outputs.
    out_sample_rate : int
        Desired output sample rate.
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

def generate_spectrograms(dataset_dir, output_dir, sequence_len, rate, n_fft=1024, n_hop=256, augment_data=False, save_plots=False):
    """ 
    Generate spectrograms (via stft) on dataset of audio data.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset.
    output_dir : str
        Directory to store outputs.
    sequence_len : int
        Length of output audio data.
    rate : int
        Sample rate out input audio data.
    n_fft : int, optional
        Size of the FFT to generate spectrograms.
    n_hop : int, optional
        Hop size for FFT.
    augment_data : bool, optional
        Generate augmented (stretched and shifted) audio.
    """

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
   
    IRs, sample_names = load_data(dataset_dir, split=False)

    specs_generated = 0
    n_specs = len(IRs)

    for idx in range(len(IRs)):
        audio = np.reshape(IRs[idx], (IRs[idx].shape[0],))

        if augment_data:
            
            augmented_audio = augment_audio(audio, 16000, 
                                            stretch_factors=[0.80, 0.90, 1.10, 1.20], 
                                            shift_factors=[-2, -1, 1, 2])
            n_specs = len(IRs) * (len(stretch_factors) + len(shift_factors))                                
            for augment in augmented_audio:
                S = librosa.stft(augment, n_fft=n_fft, hop_length=n_hop, center=True)
                power_spectra = np.abs(S)**2
                log_power_spectra = librosa.power_to_db(power_spectra)
                _min = np.amin(log_power_spectra)
                print(_min)
                print(_max)
                _max = np.amax(log_power_spectra)
                normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
                filename = f"ir_{sample_names[idx]}_{specs_generated+1}"
                np.savetxt(os.path.join(output_dir, filename + ".txt"), normalized_log_power_spectra)
                specs_generated += 1

                if save_plots:
                    if not os.path.isdir("spect_plots"):
                        os.makedirs("spect_plots")
                    plot_spectrograms(log_power_spectra, normalized_log_power_spectra, 
                                      16000, filename + ".png", "spect_plots")
        
        S = librosa.stft(audio, n_fft=n_fft, hop_length=n_hop, center=True)
        power_spectra = np.abs(S)**2
        log_power_spectra = librosa.power_to_db(power_spectra)
        _min = np.amin(log_power_spectra)
        _max = np.amax(log_power_spectra)
        normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
        filename = f"ir_{sample_names[idx]}_{specs_generated+1}"
        np.savetxt(os.path.join(output_dir, filename + ".txt"), normalized_log_power_spectra)
        specs_generated += 1

        if save_plots:
            if not os.path.isdir("spect_plots"):
                os.makedirs("spect_plots")
            plot_spectrograms(normalized_log_power_spectra, 16000, filename + ".png", "spect_plots")

        sys.stdout.write(f"* Computed {specs_generated}/{n_specs} RIR spectrograms\r")
        sys.stdout.flush()

def plot_spectrograms(log_power_spectra, rate, filename, output_dir):
    """ 
    Save log-power and normalized log-power specotrgrams to file.

    Parameters
    ----------
    log_power_spectra : ndarray
        Comptued Log-Power spectra.
    normalized_power_spectra : ndarray
        Computed normalized (between 0 and 1) Log-Power spectra.
    rate : int
        Sample rate out input audio data.
    filename : str
        Output filename for generated plot.
    output_dir : str
        Directory to save generated plot. (must exist)
    """

    plt.figure()
    librosa.display.specshow(log_power_spectra, sr=rate*2, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Power spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close('all')

def augment_audio(data, rate, stretch_factors=[], shift_factors=[]):
    """ 
    Perform data augmentation (stretching and pitch shifting) on input data.

    Parameters
    ----------
    data : ndarray
        Monophonic autio data.
    rate : int
        Sample rate out input audio data.
    stretch_factors : list of floats
        List of factors to stretch the input data by.
    shift_factors : list of floats
        List of factors to pitch the input data by.
    """
    augmented_audio = []

    # stretch audio
    for stretch_factor in stretch_factors:
        sys.stdout.write("* Strecthing audio by {}...\r".format(stretch_factor))
        sys.stdout.flush()
        augmented_audio.append(librosa.effects.time_stretch(data, stretch_factor))
    
    # pitch shift audio
    for shift_factor in shift_factors:
        sys.stdout.write("* Pitching audio by {}...\r".format(shift_factor))
        sys.stdout.flush()
        augmented_audio.append(librosa.effects.pitch_shift(data, rate, shift_factor))

    return augmented_audio

def clean_dataset(dataset_dir, reject_dir, min_len=0.5, max_len=6.0):
    """ 
    Analyze the dataset and remove samples that do not fall
    within the supplied limits.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset.
    reject_dir : str
        Directory to move rejected samples to.
    min_len : float
        Minimum valid length of audio in seconds.
    max_len : float
        Maximum valid length of audio in seconds.
    """    

    if not os.path.isdir(reject_dir):
        os.makedirs(reject_dir)

    n_rejected = 0

    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.wav"))):
        data, rate = sf.read(sample, always_2d=True)

        length = data.shape[0] / 16000.
        filename = os.path.basename(sample)

        if length < min_len: # move IRs that are too short
            shutil.move(sample, os.path.join(reject_dir, filename))
            n_rejected += 1
        elif length > max_len: # move but also shorten long IRs
            shutil.move(sample, os.path.join(reject_dir, filename))
            shortened_data = data[:int(max_len*16000),:]
            sf.write(os.path.join(dataset_dir, "short_" + filename), shortened_data, 16000)
            n_rejected += 1

        sys.stdout.write("* Analyzed {0:4} RIRs | {1:4} accepted | {2:4} rejected\r".format(idx+1, idx+1-n_rejected, n_rejected))
        sys.stdout.flush()

    print("* Analyzed {0:4} RIRs | {1:4} accepted | {2:4} rejected\r".format(idx+1, idx+1-n_rejected, n_rejected))
    print("Cleaning complete.")

def analyze_dataset(dataset_dir,):
    """ 
    Analyze and calculate relevant statistics on the dataset.

    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset.
    """
    analysis = []

    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.wav"))):
        data, rate = sf.read(sample, always_2d=True)

        analysis.append(
            {'Name' : os.path.basename(sample).replace('.wav', ''),
             'Channels' : data.shape[1],
             'Samples' : data.shape[0],
             'Length' : data.shape[0] / 16000.,
             'RMSE' : np.mean(librosa.feature.rmse(y=data.T)),
             'Flatness' : np.mean(librosa.feature.spectral_flatness(y=data[:,0]))})

        sys.stdout.write("* Analyzed {} RIRs\r".format(idx+1))
        sys.stdout.flush()

    with open('complete_stats.csv', 'w', newline='') as csvfile:
        c = csv.DictWriter(csvfile, fieldnames=analysis[0].keys())
        c.writeheader()
        c.writerows(analysis)        

    channels = [sample['Channels'] for sample in analysis]
    samples = [sample['Samples'] for sample in analysis]
    length = [sample['Length'] for sample in analysis]
    rmse = [sample['RMSE'] for sample in analysis]
    flatness = [sample['Flatness'] for sample in analysis]

    mean_channels = np.mean(channels)
    min_channels = np.min(channels)
    max_channels = np.max(channels)

    mean_samples = np.mean(samples)
    min_samples = np.min(samples)
    max_samples = np.max(samples)

    mean_length = np.mean(length)
    min_length = np.min(length)
    max_length = np.max(length)

    with open('collected_stats.csv', 'w', newline='') as csvfile:
        c = csv.DictWriter(csvfile, fieldnames=['Metric', 'Min', 'Max', 'Mean'])
        c.writeheader()
        c.writerow({'Metric' : 'Channels', 'Min' : min_channels, 'Max' : max_channels, 'Mean' : mean_channels})
        c.writerow({'Metric' : 'Samples', 'Min' : min_samples, 'Max' : max_samples, 'Mean' : mean_samples})
        c.writerow({'Metric' : 'Length', 'Min' : min_length, 'Max' : max_length, 'Mean' : mean_length})  


def generate_report(report_dir, r, msg=''):
    with open(os.path.join(report_dir, "report_summary.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write(f"Start time: {r['start time']}\n")
        results.write(f"End time:   {r['end time']}\n")
        results.write(f"Runtime:    {r['end time'] - r['start time']}\n\n")
        results.write("--- MESSAGE ---\n")
        results.write(f"{msg}\n\n")
        results.write("--- MSE RESULTS ---\n")
        val_losses = []
        for fold, track_id in zip(r["history"], r["index list"]):
            results.write(f"* Track {track_id}\n")
            results.write("    train   |  val\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(fold.history["loss"], 
                                                        fold.history["val_loss"])):
                results.write("{0}: {1:0.6f}   {2:0.6f}\n".format(epoch+1, 
                                                train_loss, val_loss))
            val_losses.append(val_loss)
            results.write("\n")
        final_loss = np.mean(val_losses)
        results.write(f"Avg. val loss: {0:0.6f}\n".format(final_loss))
        results.write("\n--- TRAINING DETAILS ---\n")
        results.write(f"Batch size:  {r['batch size']}\n")
        results.write(f"Epochs:      {r['epochs']}\n")
        results.write(f"Input shape: {r['input shape']}\n")
        results.write(f"Model type:  {r['model type']}\n")
        results.write(f"Folds:       {r['folds']:d}\n")
        results.write(f"Learning:    {r['learning rate']:f}\n")
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        r["model"].summary(print_fn=lambda x: results.write(x + '\n'))
        
        val_loss = [fold.history['val_loss'] for fold in r['history']]
        train_loss = [fold.history['loss'] for fold in r['history']]

        history = {'val loss' : val_loss,
                   'train loss' : train_loss,
                   'final loss' : final_loss}

        pickle.dump(history, open(os.path.join(report_dir, 
                    "history.pkl"), "wb"), protocol=2)
        return final_loss
    
#load_spectrograms('spectrograms', n_samples=100)
generate_spectrograms('data_16k', 'spectrograms', 66304, 16e3, augment_data=False, save_plots=True)
#analyze_dataset('data_16k')
#clean_dataset('data_16k', 'data_16k_rejected')