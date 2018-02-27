import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import DLA

# Matplotlib needs to know where ffmpeg is located.
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin'

PARTICLES = 200
N = 60
M = 45
STICKING_PROBABILITY = 0.1

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

movie = animation.FuncAnimation(fig, update, frames=len(images), repeat=True, interval=50, blit=True)
plt.show()

# Create the animation
# NOTE: You must have FFMPEG installed and in your path to actually create the
# animation
#writer = animation.FFMpegWriter(fps=30, codec=None, bitrate=None, extra_args=None, metadata=None)
#movie.save('basic_animation.mp4', writer=writer)