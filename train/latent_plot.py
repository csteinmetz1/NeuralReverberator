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
import shutil

def generate_frames(path):

    shutil.rmtree("latent_plot/3d")
    shutil.rmtree("latent_plot/specgrams")
    shutil.rmtree("latent_plot/frames")

    os.makedirs("latent_plot/3d")
    os.makedirs("latent_plot/specgrams")
    os.makedirs("latent_plot/frames")

    decoder = load_model("models/decoder.hdf5")

    steps = 100

    if   path == 'line':
        xdata = np.linspace(-2.0, 2.0, steps)
        ydata = np.linspace(2.0, -2.0, steps)
        zdata = np.linspace(-2.0, 2.0, steps) 
    elif path == 'circle':
        R = 1.5#np.linspace(-1.5, 1.5, steps)
        h = 0.0 * np.ones(100)
        u = np.linspace(0,  2*np.pi, steps)

        xdata = R * np.cos(u)
        ydata = R * np.sin(u)
        zdata = R * np.sin(u)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for idx, p in enumerate(zip(xdata, ydata, zdata)):

        # latent space plot
        ax.scatter3D([p[0]], [p[1]], [p[2]], c='r')
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(-2.0, 2.0)
        plot_filepath = "latent_plot/3d/{0}_{1:+0.3f}_{2:+0.3f}_{3:+0.3f}".format(idx, p[0], p[1], p[2])
        fig.tight_layout()
        plt.savefig(plot_filepath + ".png")
        #plt.close()

        # spectrogram plot
        print("{:04d} | z = [ {:+0.3f} {:+0.3f} {:+0.3f} ]".format(idx, p[0], p[1], p[2]))
        z = np.reshape(np.array([p[0], p[1], p[2]]), (1, 1, 1, 3)) # think i want to fix this in my model
        specgram_filepath = "latent_plot/specgrams/{0}_{1:+0.3f}_{2:+0.3f}_{3:+0.3f}".format(idx, p[0], p[1], p[2])
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
    imageio.mimsave('latent_plot/traverse_circle_tilt.gif', frames)

def generate_video():
    os.system("ffmpeg -r 25 -i latent_plot/frames/frame_%02d.png -vcodec mpeg4 -b 2000k -y latent_plot/movie.mp4")

if __name__ == "__main__":
    #generate_frames(path="circle")
    #generate_animation()
    generate_video()