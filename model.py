import sys
import keras
from keras import layers
from keras import optimizers
from keras.models import Model
import keras.backend as K

def build_vae_model(n_samples, n_ch):
	n_ch = 1
	ir_shape = (n_samples, n_ch)
	batch_size = 10
	latent_dim = 2

	input_ir = layers.Input(shape=ir_shape)

	# build the encoder
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(input_ir)
	x = layers.MaxPool1D(2, padding='same')(x)
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(x)
	x = layers.MaxPool1D(2, padding='same')(x)
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(x)
	x = layers.MaxPool1D(2, padding='same')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(8, activation='relu')(x)
	
	z_mean = layers.Dense(latent_dim)(x)
	z_log_var = layers.Dense(latent_dim)(x)

	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), 
								mean=0., stddev=1.)

		return z_mean + K.exp(z_log_var) * epsilon

	z = layers.Lambda(sampling)([z_mean, z_log_var])

	encoder = Model(input_ir, z)
	encoder.summary()

	# build the decoder
	input_z = layers.Input(shape=K.int_shape(z)[1:])

	x = layers.Dense(8, activation='relu')(input_z)
	x = layers.Dense(n_samples * 4, activation='relu')(x)
	x = layers.Reshape((int(n_samples/8), 32))(x)
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(x)
	x = layers.UpSampling1D(2)(x)
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(x)
	x = layers.UpSampling1D(2)(x)
	x = layers.Conv1D(32, 16, padding='same', activation='relu')(x)
	x = layers.UpSampling1D(2)(x)
	x = layers.Conv1D(n_ch, 16, padding='same', activation='tanh')(x)

	decoder = Model(input_z, x)
	decoder.summary()

	z_decoded = decoder(z)

	class CustomVariationalLayer(keras.layers.Layer):

		def vae_loss(self, x, z_decoded):
			x = K.flatten(x)
			z_decoded = K.flatten(z_decoded)
			xent_loss = keras.metrics.mean_squared_error(x, z_decoded)
			kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)	
			#kl_loss = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)

			return xent_loss + kl_loss

		def call(self, inputs):
			x = inputs[0]
			z_decoded = inputs[1]
			loss = self.vae_loss(x, z_decoded)
			self.add_loss(loss, inputs=inputs)
			return x

	y = CustomVariationalLayer()([input_ir, z_decoded])

	vae = Model(input_ir, y)
	vae.compile(optimizer=optimizers.Adam(), loss=None)
	vae.summary()

	return vae