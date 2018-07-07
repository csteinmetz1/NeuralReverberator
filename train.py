from util import load_data, load_spectrograms, GenerateIRs
from model import build_vae_model, build_spectral_ae
from keras.callbacks import LambdaCallback

# hyperparameters
epochs = 1
batch_size = 10
latent_dim = 2

# input
rate = 16000

# build the model
encoder, decoder, autoencoder = build_spectral_ae((512, 256, 1))

# load the data
#x_train, x_test = load_data('data_16k', sequence_len, train_split=0.90)
x_train, x_test = load_spectrograms('spectrograms', n_samples=3000)

# setup callback to generate IR samples
#audio_callback = GenerateIRs(15, rate, batch_size, latent_dim, decoder)

# train the thing
autoencoder.fit(x=x_train, y=x_train,
				shuffle=True,
				epochs=epochs,
				batch_size=batch_size,
				validation_data=(x_test, x_test))

decoder.save("models/decoder.hdf5")