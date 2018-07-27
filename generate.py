import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from keras.models import load_model
from util import ispecgram, fix_specgram_shape

import matplotlib as mpl  
mpl.use('agg')
import matplotlib.pyplot as plt

def generate_z(encoder, spec):
    """
    Determine the latent representation of a spectrogram.

    Args:
        encoder (obj): trained Keras encoder network.
        spec (ndarray): spectrogram of shape (freqs, time).
    
    Returns:
        z (ndarray): latent vector of shape (1, 1, 1, 3)
    """
    # fix shape (may be longer or shorter)
    spec = fix_specgram_shape(spec, (513, 256))

    # reshape for input to the encoder
    spec = np.reshape(spec, (1, spec.shape[1], spec.shape[2], 1))

    # predict embedding to latent vector z
    z = encoder.predict(spec)

    return z

def generate_specgram(decoder, z):
    """
    Generate a spectrogram from a latent representation.

    Args:
        decoder (obj): trained Keras decoder network.
        z (ndarray): latent vector of shape (1, 1, 1, 3).
    Returns:
        spec (ndarray): spectrogram of shape (freqs, time).
    """
    spec = decoder.predict(z) # predict spectrogram
    spec = np.reshape(spec, (spec.shape[1], spec.shape[2]))
    return spec

def audio_from_specgram(spec, rate, output):
    """
    Reconstruct audio and save it to file.

    Args:
        spec (ndarray): spectrogram of shape (freqs, time).
        rate (int): sample rate of input audio.
        output (str): path to output file.
    """
    spec = np.reshape(spec, (spec.shape[0], spec.shape[1], 1)) # reshape
    audio = ispecgram(spec, n_fft=1024, hop_length=256, mag_only=True, num_iters=1000)
    sf.write(output + '.wav', audio, rate) 

def plot_from_specgram(spec, rate, output):
    """
    Plot a spectrogram and save it to file.

    Args:
        spec (ndarray): spectrogram of shape (freqs, time).
        rate (int): sample rate of input audio.
        output (str): path to output file.
    """
    plt.figure()
    librosa.display.specshow(spec, sr=rate*2, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output + '.png')
    plt.close()
    

if __name__ == '__main__':
    # loaded trained models
    encoder = load_model("models/encoder.hdf5")
    decoder = load_model("models/decoder.hdf5")

    # generate audio and plots over latent space
    idx = 0
    for a in np.linspace(-1, 1, num=5):
        for b in np.linspace(-1, 1, num=5):
            for c in np.linspace(-1, 1, num=5):
                print("{:03d} | z = {:+0.3f} {:+0.3f} {:+0.3f}".format(idx, a, b, c))
                z = np.reshape(np.array([a, b, c]), (1, 1, 1, 3)) # think i want to fix this in my model
                filename = "_".join(["({:+0.3f})".format(dim) for dim in np.reshape(z, (3))])
                filename = "{:03d}_{}".format(idx, filename)
                filepath = os.path.join('latent', filename)
                spec = generate_specgram(decoder, z)
                audio_from_specgram(spec, 16000, filepath)
                plot_from_specgram(spec, 16000, filepath)
                idx += 1