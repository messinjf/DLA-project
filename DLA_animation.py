import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy.ndimage as ndimage

import DLA



PARTICLES = 100
N = 200
STICKING_PROBABILITY = 1

def prune_empty_space(images):
    """ Our structure is small compared to the actual grid. This is due to the
    fact that our spawning circle must also be inside the matrix, which is now
    2 times the max radius of the structure. This method solves this issue
    by removing rows and columns that are outside of the max radius of the
    final structure. """
    final_image = images[-1]
    max_radius = int(np.ceil(DLA.get_max_radius_of_structure(final_image)))
    center_i = int(final_image.shape[0]//2)
    center_j = int(final_image.shape[1]//2)
    
    for k in range(len(images)):
        images[k] = images[k][center_i-max_radius:center_i+max_radius,center_j-max_radius:center_j+max_radius]
    
    return images

images = DLA.DLA(PARTICLES, N, STICKING_PROBABILITY)
images = prune_empty_space(images)
#images = [ndimage.gaussian_filter(image, sigma=(3, 3), order=0) for image in images]

# Initialize figure
fig = plt.figure()
plt.title("Particles = {}; N = {}; prob = {}".format(PARTICLES, N, STICKING_PROBABILITY))

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
# It's actually really painful to get ffmpeg to work properly. I found that
# the best way to do it was to install it through anaconda with the following
# command:
#           conda install -c conda-forge ffmpeg
#writer = animation.FFMpegWriter(fps=30, codec=None, bitrate=None, extra_args=None, metadata=None)
#movie.save('gaussian_filter_snowflake_test.mp4', writer=writer)