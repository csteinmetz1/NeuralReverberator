import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
from keras.models import load_model
from util import ispecgram, fix_specgram_shape, plot_specgrams
from generate import generate_z, generate_specgram

import matplotlib as mpl  
mpl.use('agg')
import matplotlib.pyplot as plt

# loaded trained models
encoder = load_model("models/encoder.hdf5")
decoder = load_model("models/decoder.hdf5")

spec = np.loadtxt('spectrograms/ir_00x00y_16000_5.txt')
spec = fix_specgram_shape(spec, (513, 128))
spec = np.reshape(spec, (513, 128, 1))
_z = generate_z(encoder, spec)
print('z = {}'.format(_z))

out_spec = generate_specgram(decoder, _z)
print(out_spec.shape)
plot_specgrams(out_spec, 16000, 'ir_00x00y_16000_5.png', './')
