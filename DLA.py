import numpy as np
import matplotlib.pyplot as plt

def random_direction(dimension):
    """
    Return a random signed unit vector.
    input: D, the dimension of the unit vector to
            be created.
    output: An numpy array of size D that is all zeros
            except for one random index, which will
            either be 1 or -1 (chosen randomly).
    """
    direction = np.zeros(dimension)
    idx = np.random.randint(0, dimension)
    direction[idx] = np.sign(np.random.uniform(-1.0,1.0))
    return direction

def random_valid_direction(point, matrix):
    """ Will return a random direction that will not lead the
    particle into another particle. """
    x = int(point[0])
    y = int(point[1])
    choices = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            neighbor = np.array([x + i, y + j])
            if ((is_inbound(neighbor, matrix))  # make sure neighbor is inbounds
                    and abs(i)+abs(j) < 2       # don't check NE, SE, SW, NW corners
                    and abs(i)+abs(j) > 0       # don't allow stationary movement
                    and matrix[neighbor[0]][neighbor[1]] == 0): 
                choices.append(np.array([i,j]))
    # No neighbor found so return false
    if(len(choices) == 0):
        print("Warning: created particle was trapped and had no available movement.")
        return np.zeros(2)
    idx = np.random.randint(len(choices))
    return choices[idx]

def is_inbound(point, matrix):
    """ Determines if the specified point is inbounds of the
    given matrix. """
    x = int(point[0])
    y = int(point[1])
    return (x >= 1
            and x < matrix.shape[1]-1
            and y >= 1
            and y < matrix.shape[0]-1)


def has_neighbor(point, matrix):
    """ Return True if the point has a particle nearby. A particle is
    considered nearby if it is 1 space north, east, south, or west of
    the current point. Also checks if there is already a particle on the
    point."""
    x = int(point[0])
    y = int(point[1])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            neighbor = np.array([x + i, y + j])
            if ((is_inbound(neighbor, matrix))  # make sure neighbor is inbounds
                    and abs(i)+abs(j) < 2       # don't check NE, SE, SW, NW corners
                    and matrix[neighbor[0]][neighbor[1]] == 1):
                return True
    # No neighbor found so return false
    return False

  
def random_normal_vector():
    """ returns a random 2D unit vector. """
    direction = np.array([1.0, 0.0])
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    R = np.zeros((2,2))
    R[0,0] = np.cos(theta)
    R[0,1] = -np.sin(theta)
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    return np.dot(R, direction)

def DLA(particles, N=40, M=16, sticking_probablity=1.0):
    matrix = np.zeros((N,N))
    center = np.array([N//2, N//2])
    
    # Counts the number of particles that have sticked to the seed
    numberOfSuccessfulSticks = 0
    
    # List that keeps track of all previous iterations of the matrix so that
    # an animation can be created later.
    images = []
    
    #Initalize the seed (currently a 2 by 2 blob at the center)
    for i in range(2):
        for j in range(2):
            matrix[center[0]+i, center[1]+j] = 1
            
    images.append(np.copy(matrix))
    
    while(numberOfSuccessfulSticks < particles):
        
        # Initialize a random walker at a position around a circle of diameter
        # M. Note that this is different from the book where the boundary of a
        # square is chosen instead.
        random_walker = np.round(center + (M / 2.0) * random_normal_vector())
        
        # Deal with the case where the walker spawns on an already occupied
        # space
        while (matrix[int(random_walker[0]), int(random_walker[1])] == 1):
            random_walker = np.round(center + (M / 2.0) * random_normal_vector())
        
        while(is_inbound(random_walker, matrix)):
            #random_walker += random_direction(2)
            random_walker += random_valid_direction(random_walker, matrix)
            
            
            if(has_neighbor(random_walker, matrix)):
                # The particle has neighbors, so check to see if the particle
                # sticks.
                if(np.random.rand() <= sticking_probablity):
                    # particle has successfully sticked to the seed, so update
                    # the matrix and add the matrix to the list of images
                    x = random_walker[0]
                    y = random_walker[1]
                    matrix[int(x),int(y)] = 1
                    numberOfSuccessfulSticks += 1
                    images.append(np.copy(matrix))
                    break
       
    return images

if __name__ == "__main__":
    images = DLA(100)
    plt.imshow(images[-1])
    plt.show()
    