import os
import generate as gen
from keras.models import load_model
from PIL import Image
import matplotlib as mpl  
#mpl.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import imageio
import glob

def generate_frames():

    if not os.path.isdir("latent_plot"):
        os.makedirs("latent_plot/3d")
        os.makedirs("latent_plot/specgrams")
        os.makedirs("latent_plot/frames")

    decoder = load_model("models/decoder.hdf5")

    for idx, x in enumerate(np.linspace(-2.0, 2.0, 100)):

        # latent space plot
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D([x], [x], [x], c='r')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(-2.0, 2.0)
        plot_filepath = "latent_plot/3d/{0}_{1:+0.3f}_{2:+0.3f}_{3:+0.3f}".format(idx, x, x, x)
        fig.tight_layout()
        plt.savefig(plot_filepath + ".png")
        plt.close()

        # spectrogram plot
        print("{:04d} | z = [ {:+0.3f} {:+0.3f} {:+0.3f} ]".format(idx, x, x, x))
        z = np.reshape(np.array([x, x, x]), (1, 1, 1, 3)) # think i want to fix this in my model
        specgram_filepath = "latent_plot/specgrams/{0}_{1:+0.3f}_{2:+0.3f}_{3:+0.3f}".format(idx, x, x, x)
        spec = gen.generate_specgram(decoder, z)
        gen.plot_from_specgram(np.abs(spec), 16000, specgram_filepath)

        # Combine plots
        result = Image.new("RGB", (1280, 480))
        files = [plot_filepath + ".png", specgram_filepath + ".png"]
        for index, filepath in enumerate(files):
            img = Image.open(filepath)
            x = index * 640
            y = 0
            w, h = img.size
            result.paste(img, (x, y, x + w, y + h))

        result.save("latent_plot/frames/frame_{:02d}.png".format(idx))

def generate_animation():

    # create animated gif
    frames = []
    for frame in glob.glob("latent_plot/frames/*.png"):
        print(frame)
        frames.append(imageio.imread(frame))
    imageio.mimsave('latent_plot/traverse.gif', frames)

if __name__ == "__main__":
    #generate_frames()
    generate_animation()