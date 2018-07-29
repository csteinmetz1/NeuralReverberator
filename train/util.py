import os
import sys
import csv
import glob
import librosa
import pickle
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

def load_specgrams(dataset_dir, spec_shape, train_split=0.80, n_samples=None):
    """
    Utility function to load spectogram data.

    Args:
        dataset_dir (str): Directory containing the dataset.
        spec_shape (tuple) : Shape of spectrograms to be loaded (freqs, time)
        train_split (float, optional): Fraction of the data to return as training samples.
        n_samples (int, optional): Number of total dataset examples to load. 
            (Deafults to full size of the dataset)

    Returns:
        x_train (ndarray): Training set (samples, freqs, time).
        x_test (ndarray): Testing set (samples, freqs, time).
    """
    if n_samples is None: # set number of samples to full dataset
        n_samples = len(glob.glob(os.path.join(dataset_dir, "*.txt")))

    x = [] # list to hold spectrograms
    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.txt"))):
        if idx < n_samples:
            s = np.loadtxt(sample)
            out = fix_specgram_shape(s, spec_shape)
            x.append(out) # create list of spectrograms
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

    Args:
        dataset_dir (str): Directory containing the dataset.
        train_split (float, optional): Fraction of the data to return as training samples.

    Returns:
        x_train (ndarray): Training examples with shape (examples, audio samples).
        x_test (ndarray): Testing examples with shape (examples, audio samples).	
    """
    IRs = [] # list to hold audio data
    sample_names = [] # temp list - delete this later - maybe not?
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

    Args:
        dataset_dir (str): Directory containing the dataset.
        output_dir (str): Directory to store outputs.
        out_sample_rate (int): Desired output sample rate.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    rir_list = glob.glob(os.path.join(dataset_dir, "*.wav"))
    for idx, sample in enumerate(tqdm(iterable=rir_list, desc="Converting sample rate", ncols=100)):
        filename = os.path.basename(sample).split('.')[0]
        out_filepath = os.path.join(output_dir, "{0}_{1}.wav".format(filename, out_sample_rate))
        subprocess.call("""sox "{0}" -r {1} "{2}" """.format(sample, out_sample_rate, out_filepath), shell=True, stderr=subprocess.DEVNULL)

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

def generate_specgrams(dataset_dir, 
                       output_dir, 
                       sequence_len, 
                       rate=16000, 
                       n_fft=1024, 
                       n_hop=256, 
                       augment_data=False, 
                       save_plots=False):
    """ 
    Generate spectrograms (via stft) on dataset of audio data.

    Args:
        dataset_dir (str): Directory containing the dataset.
        output_dir (str): Directory to store outputs.
        sequence_len (int): Length of output audio data.
        rate (int, optional): Sample rate out input audio data.
        n_fft (int, optional): Size of the FFT to generate spectrograms.
        n_hop (int, optional): Hop size for FFT.
        augment_data (bool, optional): Generate augmented (stretched and shifted) audio.
        save_plot (bool, optional): Generate plots of spectrograms
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
                _max = np.amax(log_power_spectra)
                normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
                filename = f"ir_{sample_names[idx]}_{specs_generated+1}"
                np.savetxt(os.path.join(output_dir, filename + ".txt"), normalized_log_power_spectra)
                specs_generated += 1

                if save_plots:
                    if not os.path.isdir("spect_plots"):
                        os.makedirs("spect_plots")
                    plot_specgrams(log_power_spectra, normalized_log_power_spectra, 
                                      16000, filename + ".png", "spect_plots")
        
        S = librosa.stft(audio, n_fft=n_fft, hop_length=n_hop, center=True)
        power_spectra = np.abs(S)**2
        log_power_spectra = librosa.power_to_db(power_spectra)
        _min = np.amin(log_power_spectra)
        _max = np.amax(log_power_spectra)
        if _min == _max:
            print(f"divide by zero in {filename}")
        else:
            normalized_log_power_spectra = (log_power_spectra - _min) / (_max - _min)
            filename = f"ir_{sample_names[idx]}_{specs_generated+1}"
            np.savetxt(os.path.join(output_dir, filename + ".txt"), normalized_log_power_spectra)
            specs_generated += 1

            if save_plots:
                if not os.path.isdir("spect_plots"):
                    os.makedirs("spect_plots")
                plot_specgrams(normalized_log_power_spectra, 16000, filename + ".png", "spect_plots")

        sys.stdout.write(f"* Computed {specs_generated}/{n_specs} RIR spectrograms\r")
        sys.stdout.flush()

def plot_specgrams(log_power_spectra, rate, filename, output_dir):
    """ 
    Save log-power and normalized log-power specotrgrams to file.

    Args:
        log_power_spectra (ndarray): Comptued Log-Power spectra.
        rate (int): Sample rate of input audio data.
        filename (str): Output filename for generated plot.
        output_dir (str): Directory to save generated plot.
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

    Args:
        data (ndarray): Monophonic autio data.
        rate (int): Sample rate out input audio data.
        stretch_factors (list of floats): List of factors to stretch the input data by.
        shift_factors (list of floats): List of factors to pitch the input data by.
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

    Args:
        dataset_dir (str): Directory containing the dataset.
        reject_dir (str): Directory to move rejected samples to.
        min_len (float, optional): Minimum valid length of audio in seconds.
        max_len (float, optional): Maximum valid length of audio in seconds.
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

def check_spectrograms_for_nans(dataset_dir):
    """
    Utility function to determine if dataset contains samples with NaNs.

    Args:
        dataset_dir (str): Directory containing the dataset.
    Returns:
        nans (bool): Boolean value indicating if NaNs are present. 
    """
    n_samples = len(glob.glob(os.path.join(dataset_dir, "*.txt")))
    n_nans = 0

    for idx, sample in enumerate(glob.glob(os.path.join(dataset_dir, "*.txt"))):
        s = np.loadtxt(sample)

        analysis = np.isnan(s)

        if np.any(analysis):
            n_nans += 1

        sys.stdout.write(f"* Checked {idx+1}/{n_samples} RIR spectrograms | Found {n_nans} NaNs\r")
        sys.stdout.flush()

def fix_specgram_shape(spec, shape):
    """Fix spectrogram shape to user specified size.
    Args:
        spec: 2D spectrogram [freqs, time].
        shape: 2D output spectrogram shape [freqs, time].
    Returns:
        fixed_spec: fixed 2D output spectrogram [freqs, time].
    """
    if spec.shape[1] < shape[1]: # pad the input to be of shape (513, 256)
        out = np.zeros(shape)
        out[:spec.shape[0],:spec.shape[1]] = spec
    else: # crop the input to be of shape (513, 256)
        out = spec[:,:shape[1]]
            
    return out

#---------------------------------------------------------------------------------
# Analysis, plotting, and report generation
#---------------------------------------------------------------------------------

def analyze_dataset(dataset_dir,):
    """ 
    Analyze and calculate relevant statistics on the dataset.

    Args:
        dataset_dir (str): Directory containing the dataset.
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

def generate_report(r, msg='', root_report_dir='reports'):
    """ 
    Collect training results into a nicely formatted txt report.

    Args:
        r (dict) : Training details and results.

        r = {'start_time' : start_time,
             'end_time' : end_time,
             'history' : history.history,
             'batch_size' : batch_size,
             'epochs' : epochs,
             'learning_rate' : learning_rate,
             'latent_dim' : latent_dim,
             'n_filters' : n_filters,
             'input_shape' : input_shape,
             'rate' : rate,
             'n_samples' : n_samples,
             'encoder' : e,
             'decoder' : d,
             'autoencoder' : ae}

        msg (str, optional): Optional message from user when running experiment.
        root_report_dir (str, optional): Directory for reports to be stored.
    """
    t = r['start_time'] # end time
    report_dir = f"train_{t.year:04}_{t.month:02}_{t.day:02}__{t.hour:02}-{t.minute:02}"

    # create report directory 
    if not os.path.isdir(os.path.join(root_report_dir, report_dir)):
        os.makedirs(os.path.join(root_report_dir, report_dir))

    with open(os.path.join(root_report_dir, report_dir, "report.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write(f"Start time: {r['start_time']}\n")
        results.write(f"End time:   {r['end_time']}\n")
        results.write(f"Runtime:    {r['end_time'] - r['start_time']}\n\n")
        results.write("--- MESSAGE ---\n")
        results.write(f"{msg}\n\n")
        results.write("--- MSE RESULTS ---\n")
        results.write("epochs | train   |  val\n")
        for epoch, (train_loss, val_loss) in enumerate(zip(r['history']['loss'], r['history']['val_loss'])):
            results.write("   {0}:   {1:0.6f}   {2:0.6f}\n".format(epoch+1, train_loss, val_loss))
        results.write("\n--- TRAINING DETAILS ---\n")
        results.write(f"Batch size:  {r['batch_size']}\n")
        results.write(f"Epochs:      {r['epochs']}\n")
        results.write(f"Learning:    {r['learning_rate']}\n")
        results.write(f"Latent dim:  {r['latent_dim']}\n")
        results.write(f"N filters:   {r['n_filters']}\n")
        results.write(f"Input shape: {r['input_shape']}\n")
        results.write(f"N samples:   {r['n_samples']}\n")
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        r['autoencoder'].summary(print_fn=lambda x: results.write(x + '\n'))
        r['encoder'].summary(print_fn=lambda x: results.write(x + '\n'))
        r['decoder'].summary(print_fn=lambda x: results.write(x + '\n'))
    
    # save models
    r['autoencoder'].save(os.path.join(root_report_dir, report_dir, "autoencoder.hdf5"))
    r['encoder'].save(os.path.join(root_report_dir, report_dir, "encoder.hdf5"))
    r['decoder'].save(os.path.join(root_report_dir, report_dir, "decoder.hdf5"))

    # save training history for chart generation
    pickle.dump(r['history'], open(os.path.join(root_report_dir, report_dir, 
                                                "history.pkl"), "wb"), protocol=2)
    # save plots of the training loss curves
    generate_training_plots(r['history'], root_report_dir, report_dir, r['start_time'])

def generate_training_plots(history, root_report_dir, report_dir, time, plots_dir='plots'):
    """ 
    Generate plots of the training and validation losses.

    Args:
        history (dict) : Training history results.
        root_report_dir (str): Directory for reports to be stored.
        report_dir (str): Directory for the current report to be stored.
        time (str): Start time of the training cycle.
        plots_dir (str, optional): Subdirectory to store plots. 
    """
    if not os.path.isdir(os.path.join(root_report_dir, report_dir, plots_dir)):
        os.makedirs(os.path.join(root_report_dir, report_dir, plots_dir))

    # create training loss plot - over all epochs
    plt.figure(1)
    loss = history['loss']
    val_loss = history['val_loss']
    n_epochs = len(loss)

    # Summary plot (train and val)
    t = np.arange(1, n_epochs+1)
    plt.plot(t, loss, label='train loss', linewidth=0.5, color='#d73c49')
    plt.plot(t, val_loss, label='val loss', linewidth=0.5, color='#417e90')
    plt.ylabel('Training Loss (MSE)')
    plt.title(f"{time} Training Run")
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().grid(True)
    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(os.path.join(root_report_dir, report_dir, plots_dir, 
                "train_and_val_loss_summary.png"))
    plt.close('all')

    # Train plot
    t = np.arange(1, n_epochs+1)
    plt.plot(t, loss, label='train loss', linewidth=0.5, color='#d73c49')
    plt.ylabel('Training Loss (MSE)')
    plt.title(f"{time} Training Run")
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().grid(True)
    plt.savefig(os.path.join(root_report_dir, report_dir, plots_dir, 
                "train.png"))
    plt.close('all')

    # Val plot
    t = np.arange(1, n_epochs+1)
    plt.plot(t, val_loss, label='val loss', linewidth=0.5, color='#417e90')
    plt.ylabel('Validation Loss (MSE)')
    plt.title(f"{time} Training Run")
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().grid(True)
    plt.savefig(os.path.join(root_report_dir, report_dir, plots_dir, 
                "val.png"))
    plt.close('all')

    # Last 10% epochs summary
    start = int(0.9*n_epochs)
    end = n_epochs
    t = np.arange(start, end)
    plt.plot(t, loss[start:end], label='train loss', linewidth=0.5, color='#d73c49')
    plt.plot(t, val_loss[start:end], label='val loss', linewidth=0.5, color='#417e90')
    plt.ylabel('Training Loss (MSE)')
    plt.title(f"{time} Training Run")
    plt.xlabel('Epoch')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().grid(True)
    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(os.path.join(root_report_dir, report_dir, plots_dir, 
                "train_and_val_loss_summary_end.png"))
    plt.close('all')

#---------------------------------------------------------------------------------
# The methods below come from the nsynth magenta project found here:
# https://github.com/tensorflow/magenta/tree/master/magenta/models/nsynth
#---------------------------------------------------------------------------------

def inv_magphase(mag, phase_angle):
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    return mag * phase

def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
    """Iterative algorithm for phase retrival from a magnitude spectrogram.
    Args:
        mag: Magnitude spectrogram.
        phase_angle: Initial condition for phase.
        n_fft: Size of the FFT.
        hop: Stride of FFT. Defaults to n_fft/2.
        num_iters: Griffin-Lim iterations to perform.
    Returns:
        audio: 1-D array of float32 sound samples.
    """
    fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
    ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
    complex_specgram = inv_magphase(mag, phase_angle)
    for i in range(num_iters):
        audio = librosa.istft(complex_specgram, **ifft_config)
        if i != num_iters - 1:
            complex_specgram = librosa.stft(audio, **fft_config)
            _, phase = librosa.magphase(complex_specgram)
            phase_angle = np.angle(phase)
            complex_specgram = inv_magphase(mag, phase_angle)
    return audio

def ispecgram(spec,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=True,
              num_iters=1000):
    """Inverse Spectrogram using librosa.
    Args:
        spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
        n_fft: Size of the FFT.
        hop_length: Stride of FFT. Defaults to n_fft/2.
        mask: Reverse the mask of the phase derivative by the magnitude.
        log_mag: Use the logamplitude.
        re_im: Output Real and Imag. instead of logMag and dPhase.
        dphase: Use derivative of phase instead of phase.
        mag_only: Specgram contains no phase.
        num_iters: Number of griffin-lim iterations for mag_only.
    Returns:
        audio: 1-D array of sound samples. Peak normalized to 1.
    """
    if not hop_length:
        hop_length = n_fft // 2

    ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

    if mag_only:
        mag = spec[:, :, 0]
        phase_angle = np.pi * np.random.rand(*mag.shape)
    elif re_im:
        spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
    else:
        mag, p = spec[:, :, 0], spec[:, :, 1]
        if mask and log_mag:
            p /= (mag + 1e-13 * np.random.randn(*mag.shape))
        if dphase:
            # Roll up phase
            phase_angle = np.cumsum(p * np.pi, axis=1)
        else:
            phase_angle = p * np.pi

    # Magnitudes
    if log_mag:
        mag = (mag - 1.0) * 120.0
        mag = 10**(mag / 20.0)
    phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
    spec_real = mag * phase

    if mag_only:
        audio = griffin_lim(
            mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
    else:
        audio = librosa.core.istft(spec_real, **ifft_config)
    return np.squeeze((audio / audio.max()) * 0.25) # scale to -12dB peak