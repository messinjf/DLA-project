import numpy as np
import matplotlib.pyplot as plt
import time

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
                
    # No neighbor which means the particle is trapped. With the current
    # technique, this situation should never happen
    assert(len(choices) != 0)
    
    idx = np.random.randint(len(choices))
    return choices[idx]

def is_inbound(point, matrix, radius=None):
    """ Determines if the specified point is inbounds of the
    given matrix. If max_radius is not None, then it will also
    check if the point is inside the radius."""
    x = int(point[0])
    y = int(point[1])
    return (x >= 1
            and x < matrix.shape[1]-1
            and y >= 1
            and y < matrix.shape[0]-1
            and ((radius) == None
                 or (get_radius_of_point(point, matrix) <= radius))
            )


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


def get_radius_of_point(point, matrix):
    """ Return the distance of the point from the center of the matrix. """
    center = np.array([matrix.shape[0]/2, matrix.shape[1]/2])
    return np.linalg.norm(np.array(point) - center)

def get_max_radius_of_structure(matrix):
    """ Returns the maximum radius of the structure. This information
    is required in order to do the optimizations associated with Project 10
    and Project 11. """
    max_r = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            # If there is a particle at the point
            if(matrix[i][j] == 1):
                r = get_radius_of_point([i, j], matrix)
                if r > max_r:
                    max_r = r
    return max_r
  
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

def DLA(particles, N=100, sticking_probablity=1.0):
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
            
    # Save the initial configuration so that it can be played back later.
    images.append(np.copy(matrix))
    
    while(numberOfSuccessfulSticks < particles):
        
        # Determine the max radius of the structure. This information
        # is used to speed up the simulation by adjusting the launching
        # circle to be at 2 times the max radius, and to remove particles
        # that are 3 times the max radius. See project 10 for details.
        max_radius = get_max_radius_of_structure(matrix)
        launching_radius = np.round(2.0 * max_radius)
        despawn_radius = np.round(3.0 * max_radius)
        
        # We spawn a random walker at 2 times the max radius of the structure.
        # This check is to make sure that particles can't spawn outside of the
        # grid
        assert(2 * launching_radius < N)
        
        # Initialize a random walker at a position around a circle of radius
        # 2 times the max_radius.
        # Note that this is different from the book where the boundary of a
        # square is chosen instead.
        random_walker = np.round(center + launching_radius * random_normal_vector())
        
        # Should not spawn on a particle now unless we have done something wrong.
        assert(matrix[int(random_walker[0]), int(random_walker[1])] != 1)
        
        while(is_inbound(random_walker, matrix, despawn_radius)):
            
            # Get a random direction for the walker to travel in.
            random_direction = random_valid_direction(random_walker, matrix)
            
            # Optimization from Project 11. If the walker is more than 4 spaces
            # away from the structure, then we can increase the step size
            # of the random walk. See project 11 for details.
            current_radius = get_radius_of_point(random_walker, matrix)
            if current_radius > max_radius + 4:
                step_size = int(np.round(current_radius - max_radius - 2))
                assert(step_size >= 1)
                random_direction *= step_size
            
            random_walker += random_direction
            
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
    t0 = time.time()
    images = DLA(100)
    t1 = time.time()
    plt.imshow(images[-1])
    plt.show()
    print(t1-t0)
    