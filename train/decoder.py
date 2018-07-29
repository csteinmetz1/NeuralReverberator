import numpy as np
import librosa
import librosa.display
import soundfile as sf
from keras.models import load_model

from util import ispecgram
import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt

def generate_z(encoder, spectrogram):
	spectrogram = np.reshape(spectrogram, (1, spectrogram.shape[1], spectrogram.shape[2], 1))
	z = encoder.predict(spectrogram)
	return z

def generate_random_spectrogram(decoder, save_img=False, save_audio=False):
	z = np.random.randn(1,1,1,3) # create random latent vector
	generate_ir(decoder, z, save_img=save_img, save_audio=save_audio)

def generate_spectrogram(decoder, z, save_img=False, save_audio=False):

	s = decoder.predict(z) # predict spectrogram
	s = np.reshape(s, (s.shape[1], s.shape[2])) # reshape
	print(s)

	filename = "_".join(["({:0.4f})".format(dim) for dim in np.reshape(z, (3))])

	if save_img:
		plt.figure()
		librosa.display.specshow(s[:,:126], sr=16000*2, y_axis='log', x_axis='time')
		plt.colorbar(format='%+2.0f dB')
		plt.tight_layout()
		plt.savefig(filename + '.png')
	
	if save_audio:
		# invert spectrogram
		# save audio
		pass
	
	return s

encoder = load_model("reports/train_2018_07_19__00-14/encoder.hdf5")
decoder = load_model("reports/train_2018_07_19__00-14/decoder.hdf5")

s = np.loadtxt('spectrograms/ir_00x20y_16000_9.txt')
if s.shape[1] < 256: # pad the input to be of shape (513, 256)
	out = np.zeros((513, 256))
	out[:s.shape[0],:s.shape[1]] = s
else: # crop the input to be of shape (513, 256)
	out = s[:,:256]
print(out.shape)

z = generate_z(encoder, np.reshape(out, (1, out.shape[0], out.shape[1], 1)))
print(z)
s = generate_spectrogram(decoder, z, save_img=True)
s = np.reshape(s, (s.shape[0], s.shape[1], 1))

audio = ispecgram(s, n_fft=1024, hop_length=256, mag_only=True, num_iters=1000)
sf.write('output.wav', audio, 16000) 


#spectrogram = decoder.predict(z)
#print(spectrogram.shape)

#spectrogram = np.reshape(spectrogram, (512, 256))
#print(spectrogram.shape)
#print(np.zeros((256)).shape)
#spectrogram = np.concatenate((spectrogram, np.zeros(1,256)), axis=0)
#spectrogram = np.insert(spectrogram, 512, values=0, axis=0)
#print(spectrogram.shape)

#plt.figure(figsize=(12, 8))
#librosa.display.specshow(spectrogram)
#plt.colorbar(format='%+2.0f dB')
#plt.tight_layout()
#plt.show()
#data, rate = sf.read('data_16k/falkland_tennis_court_omni_16000.wav')
#data = librosa.stft(data, n_fft=1024, hop_length=256, center=True)
#data = np.abs(data)**2
#data = data * (1.0 / np.amax(data))
#print(data)
#print(np.amax(data))
 
#data = np.loadtxt('spectrograms/ir_00x20y_16000_9.txt')
#data = np.zeros((65280,))
#print(data.shape)
#data = librosa.stft(data, n_fft=1024, hop_length=256, center=True)
#print(data.shape)
#istft = griffin_lim(data, 0, 1024, 256, 10000)
#print(istft.shape)
#out = pyloudnorm.normalize.peak(istft, -3.0)
                      	