from util import load_data, GenerateIRs
from model import build_vae_model
from keras.callbacks import LambdaCallback

# hyperparameters
epochs = 10
batch_size = 10
latent_dim = 2

# input
sequence_len = 4000
rate = 16000
n_ch = 1

# build the model
encoder, decoder, vae = build_vae_model(sequence_len, n_ch)

# load the data
x_train, x_test = load_data('data_16k', sequence_len, train_split=0.90)

# setup callback to generate IR samples
audio_callback = GenerateIRs(15, rate, batch_size, latent_dim, decoder)

# train the thing
vae.fit(x=x_train, y=None,
		shuffle=True,
		epochs=10,
		batch_size=10,
		validation_data=(x_test, None),
		callbacks=[audio_callback])