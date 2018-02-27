import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import DLA

PARTICLES = 100
N = 40
M = 16
STICKING_PROBABILITY = 1.0

images = DLA.DLA(PARTICLES, N, M, STICKING_PROBABILITY)

# Initialize figure
fig = plt.figure()
plt.title("Particles = {}; N = {}; M = {}: prob = {}".format(PARTICLES, N, M, STICKING_PROBABILITY))

# Show the first image, and specify to allow for animations
im = plt.imshow(images[0], animated=True)

def update(i):
    """ Display the next image in the list. """
    im.set_array(images[i])
    return im,

movie = animation.FuncAnimation(fig, update, frames=len(images), repeat=False, interval=50, blit=True)
#plt.show()

# Create the animation
# NOTE: You must have FFMPEG installed and in your path to actually create the
# animation
writer = animation.FFMpegWriter(fps=30, codec=None, bitrate=None, extra_args=None, metadata=None)
movie.save('basic_animation.mp4', writer=writer)