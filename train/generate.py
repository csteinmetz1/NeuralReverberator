import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from keras.models import load_model
from util import ispecgram, fix_specgram_shape, load_specgrams

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
    spec_shape = (encoder.input_shape[1], encoder.input_shape[2])
    spec = fix_specgram_shape(spec, spec_shape)

    # reshape for input to the encoder
    spec = np.reshape(spec, (1, spec.shape[0], spec.shape[1], 1))

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

    # generate embeddings for all of the training spectrograms
    #z = [] # list to store embeddings
    #x_train, x_test = load_specgrams('spectrograms', (513, 128), train_split=1.0)
    #for spec in x_train:
    #    _z = generate_z(encoder, spec)
    #    print(_z)
    #    z.append(_z)

    #print('Max:', np.max(z))
    #print('Mean:', np.mean(z))
    #rint('Min:', np.min(z))

    # generate audio and plots over latent space
    idx = 0
    for a in np.linspace(-2, 2, num=10):
        for b in np.linspace(-2, 2, num=10):
            for c in np.linspace(-2, 2, num=10):
                print("{:04d} | z = [ {:+0.3f} {:+0.3f} {:+0.3f} ]".format(idx, a, b, c))
                z = np.reshape(np.array([a, b, c]), (1, 1, 1, 3)) # think i want to fix this in my model
                filename = "_".join(["({:+0.3f})".format(dim) for dim in np.reshape(z, (3))])
                filename = "{:04d}_{}".format(idx, filename)
                filepath = os.path.join('pre_compute2', filename)
                spec = generate_specgram(decoder, z)
                audio_from_specgram(spec, 16000, filepath)
                plot_from_specgram(spec, 16000, filepath)
                idx += 1

                # now perturb z to generete second channel
                e = np.array([np.random.normal(scale=0.01), np.random.normal(scale=0.01), np.random.normal(scale=0.01)])
                z = np.array([a, b, c]) - e
                print("{:04d} | z = [ {:+0.3f} {:+0.3f} {:+0.3f} ]".format(idx, z[0], z[1], z[2]))
                z = np.reshape(z, (1, 1, 1, 3)) # think i want to fix this in my model
                filename = "_".join(["({:+0.3f})".format(dim) for dim in np.reshape(z, (3))])
                filename = "{:04d}_{}".format(idx, filename)
                filepath = os.path.join('pre_compute2', filename)
                spec = generate_specgram(decoder, z)
                audio_from_specgram(spec, 16000, filepath)
                plot_from_specgram(spec, 16000, filepath)
                idx += 1