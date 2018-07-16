import pickle
import datetime
from util import load_spectrograms
from model import build_spectral_ae

# hyperparameters
epochs = 25
batch_size = 1
latent_dim = 1024
n_filters = [64, 128, 256, 1024]

# input
input_shape = (513, 256, 1)
rate = 16000

# build the model
e, d, ae = build_spectral_ae(input_shape=input_shape, 
                            latent_dim=latent_dim,
                            n_filters=n_filters)

# load the data
x_train, x_test = load_spectrograms('spectrograms', n_samples=10)

start_time = datetime.datetime.today()

# train the thing
history = ae.fit(x=x_train, y=x_train,
                shuffle=True,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test))

end_time = datetime.datetime.today()

print(f"Completed {epochs} in {end_time-start_time}.")
# generate report here

d.save("models/decoder.hdf5")