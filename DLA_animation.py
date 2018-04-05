import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import scipy.ndimage as ndimage

def animimate_images(images, title, save_name):
    # Initialize figure
    fig = plt.figure()
    plt.title(title)
    
    # Show the first image, and specify to allow for animations
    im = plt.imshow(images[0], animated=True)
    
    def update(i):
        """ Display the next image in the list. """
        im.set_array(images[i])
        return im,
    
    movie = animation.FuncAnimation(fig, update, frames=len(images), repeat=True, interval=17, blit=True)
    #plt.show()
    
    writer = animation.FFMpegWriter(fps=60, codec=None, bitrate=-1, extra_args=None, metadata=None)
    movie.save(save_name, writer=writer)