from util import load_data
from model import build_vae_model

# build the model
vae_model = build_vae_model(8000, 1)

# load the data
x_train, x_test = load_data('data_16k', 8000, train_split=0.98)

# train the thing
vae_model.fit(x=x_train, y=None,
			shuffle=True,
			epochs=10,
			batch_size=10,
			validation_data=(x_test, None))