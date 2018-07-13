from util import load_spectrograms, GenerateIRs
from model import build_spectral_ae
from keras.callbacks import LambdaCallback

# hyperparameters
epochs = 1
batch_size = 10
latent_dim = 2
n_filters = [32, 128, 256, 1024]

# input
input_shape = (513, 256, 1)
rate = 16000

# build the model
e, d, ae = build_spectral_ae(input_shape=input_shape, 
                            latent_dim=latent_dim,
                            n_filters=n_filters)

# load the data
x_train, x_test = load_spectrograms('spectrograms', n_samples=10)

# setup callback to generate IR samples
#audio_callback = GenerateIRs(15, rate, batch_size, latent_dim, decoder)

# train the thing
ae.fit(x=x_train, y=x_train,
       shuffle=True,
       epochs=epochs,
       batch_size=batch_size,
       validation_data=(x_test, x_test))

d.save("models/decoder.hdf5")