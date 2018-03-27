import numpy as np
import matplotlib.pyplot as plt
import time

DIRECTIONS = np.array([[-1,-1],[0,-1], [1,-1], [-1,0],[1,0],[-1,1], [0,1], [1,1]])
#DIRECTIONS = np.array([[1,0],[-1,0],[0,1],[0,-1]])

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

def get_direction_that_points_towards_center(vector):
    # Create the direction vector
    direction = np.zeros(2)
    
    forward_directions = []
    
    max_dot = 0
    for d in DIRECTIONS:
        dot_product = np.dot(vector, d / np.linalg.norm(d))
        if(dot_product > 0):
            forward_directions.append(d)
            if(dot_product > max_dot):
                max_dot = dot_product
                direction = d
        
    assert(len(forward_directions) > 0 and len(forward_directions) <= len(DIRECTIONS)/2)
    return direction, forward_directions

def weighted_valid_direction(point, center, X, Y, matrix):
    """ Will return a random direction that will not lead the
    particle into another particle. This direction is weighted
    and is more likely to pick towards the center. Influence
    affects the weight and should be between 0 and 1"""
    assert(X >= 0 and X <= 1)
    assert(Y >= 0 and Y <= 1)
    
    # Find the vector pointing toward the center
    center_direction = center - point
    center_direction, forward_directions = get_direction_that_points_towards_center(center_direction)
    
    # Weight for the direction pointing to the center
    other_weights = (1-X) * (1-Y) /8.0
    forward_weight = other_weights + X * (1-Y) * 5.0 / 8.0
    pointing_weight = forward_weight + Y
    # Weight for the other directions
    
    
    weights = []
    choices = []
    for direction in DIRECTIONS:
        neighbor = np.array(point + direction)
        if ((is_inbound(neighbor, matrix))  # make sure neighbor is inbounds
                and matrix[int(neighbor[0])][int(neighbor[1])] == 0):
            choices.append(np.copy(direction))
            for forward_direction in forward_directions:
                if(np.all(np.equal(forward_direction, center_direction))):
                    if(np.all(np.equal(direction, center_direction))):
                        weights.append(pointing_weight)
                        break
                    else:
                        weights.append(forward_weight)
                        break
            else:
                weights.append(other_weights) 
                
    # No neighbor which means the particle is trapped. With the current
    # technique, this situation should never happen
    if(len(choices) == 0):
        print(point)
        matrix[int(point[0]),int(point[1])] = 2
        plt.imshow(matrix)
        plt.show()
        assert(len(choices) != 0)
    
    
    #Normalize weights so that they sum up to 1
    if(not np.sum(weights) > 0):
        print(weights)
        assert(np.sum(weights) > 0)
    weights /= np.sum(weights)
    
    idx = np.random.choice(len(choices), p=weights)
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
    for direction in DIRECTIONS:
        neighbor = np.array(point + direction)
        if ((is_inbound(neighbor, matrix))  # make sure neighbor is inbounds
                and matrix[int(neighbor[0])][int(neighbor[1])] == 1):
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


def rotate(theta):
    R = np.zeros((2,2))
    R[0,0] = np.cos(theta)
    R[0,1] = -np.sin(theta)
    R[1,0] = np.sin(theta)
    R[1,1] = np.cos(theta)
    return R
  
def random_normal_vector():
    """ returns a random 2D unit vector. """
    direction = np.array([1.0, 0.0])
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    return np.dot(rotate(theta), direction)

def DLA(particles, N=100, sticking_probablity=1.0, X=0.5, Y=0.5):
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
    
    # Determine the max radius of the structure. This information
    # is used to speed up the simulation by adjusting the launching
    # circle to be at 2 times the max radius, and to remove particles
    # that are 3 times the max radius. See project 10 for details.
    max_radius = get_max_radius_of_structure(matrix)
    
    while(numberOfSuccessfulSticks < particles):
        
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
            # random_direction = random_valid_direction(random_walker, matrix)
            #Get a weighted random direction for the walker to travel in.
            random_direction = weighted_valid_direction(random_walker, center, X, Y, matrix)
            
            # Optimization from Project 11. If the walker is more than 4 spaces
            # away from the structure, then we can increase the step size
            # of the random walk. See project 11 for details.
            current_radius = get_radius_of_point(random_walker, matrix)
            # TODO: This optimization has to be modified when diagonal directions
            # are allowed as the particle can move farther than intended in one
            # step
            # current_radius /= np.sqrt(2)
#            if current_radius > max_radius + 4:
#                step_size = int(np.floor(current_radius - max_radius - 2))
#                assert(step_size >= 1)
#                random_direction = random_direction / np.linalg.norm(random_direction)
#                random_direction *= step_size
#                random_direction = np.floor(random_direction)
                #print(random_direction)
            
            random_walker += random_direction
            
            if(has_neighbor(random_walker, matrix)):
                
                # The particle has neighbors, so check to see if the particle
                # sticks.
                if(np.random.rand() <= sticking_probablity):
                    
                    # particle has successfully sticked to the seed, so update
                    # the matrix and add the matrix to the list of images
                    matrix[int(random_walker[0]),int(random_walker[1])] = 1
                    
                    # Update max_radius if new particle is outside the radius
                    # of the structure
                    walker_radius = get_radius_of_point(random_walker,matrix)
                    max_radius = max(max_radius, walker_radius)
                           
                    numberOfSuccessfulSticks += 1
                    images.append(np.copy(matrix))
                    break
       
    return images

def prune_empty_space(images):
    """ Our structure is small compared to the actual grid. This is due to the
    fact that our spawning circle must also be inside the matrix, which is now
    2 times the max radius of the structure. This method solves this issue
    by removing rows and columns that are outside of the max radius of the
    final structure. """
    final_image = images[-1]
    max_radius = int(np.ceil(get_max_radius_of_structure(final_image)))
    center_i = int(final_image.shape[0]//2)
    center_j = int(final_image.shape[1]//2)
    
    for k in range(len(images)):
        images[k] = images[k][center_i-max_radius:center_i+max_radius,center_j-max_radius:center_j+max_radius]
    
    return images

if __name__ == "__main__":
    t0 = time.time()
    N = 200
    X = 0.2
    Y = 0.2
    images = DLA(N, 200, 1, X, Y)
    images = prune_empty_space(images)
    t1 = time.time()
    title_string = "Number of particles: {}; Y: {}".format(N,Y)
    plt.title(title_string)
    plt.imshow(images[-1])
    plt.show()
    print(t1-t0)
    