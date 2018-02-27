import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

import DLA

ITERATIONS = 100
N = 40
M = 16

images = DLA.DLA(ITERATIONS, N, M)

# Initialize figure
fig = plt.figure()
plt.title("Iterations = {}; N = {}; M = {}".format(ITERATIONS, N, M))

# Show the first image, and specify to allow for animations
im = plt.imshow(images[0], animated=True)

def update(i):
    """ Display the next image in the list. """
    im.set_array(images[i])
    return im,

movie = animation.FuncAnimation(fig, update, frames=len(images), repeat=False, interval=50, blit=True)
plt.show()

# Create the animation
# NOTE: You must have FFMPEG installed and in your path to actually create the
# animation
#writer = animation.FFMpegWriter(fps=30, codec=None, bitrate=None, extra_args=None, metadata=None)
#movie.save('basic_animation.mp4', writer=writer)