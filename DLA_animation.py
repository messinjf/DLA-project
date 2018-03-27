import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy.ndimage as ndimage

import DLA

PARTICLES = 2000
N = 300
STICKING_PROBABILITY = 1
Y = 0.6

images = DLA.DLA(PARTICLES, N, STICKING_PROBABILITY)
images = DLA.prune_empty_space(images)
#images = [ndimage.gaussian_filter(image, sigma=(3, 3), order=0) for image in images]

# Initialize figure
fig = plt.figure()
plt.title("Particles = {}; Y = {}".format(PARTICLES, Y))

# Show the first image, and specify to allow for animations
im = plt.imshow(images[0], animated=True)

def update(i):
    """ Display the next image in the list. """
    im.set_array(images[i])
    return im,

movie = animation.FuncAnimation(fig, update, frames=len(images), repeat=True, interval=17, blit=True)
#plt.show()

# Create the animation
# NOTE: You must have FFMPEG installed and in your path to actually create the
# animation
# It's actually really painful to get ffmpeg to work properly. I found that
# the best way to do it was to install it through anaconda with the following
# command:
#           conda install -c conda-forge ffmpeg
writer = animation.FFMpegWriter(fps=60, codec=None, bitrate=-1, extra_args=None, metadata=None)
movie.save('city_test_particles{}_Y{}.mp4'.format(PARTICLES,int(round(Y*100))), writer=writer)