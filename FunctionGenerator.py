import numpy as np
import random
from scipy.ndimage import gaussian_filter
from collections import deque
import plotly.graph_objs as go




def generate_smooth_function(array_size=(100,100,100), fraction=.3, smoothness=10, compactness=.4):
    smoothness = int(smoothness)
    compactness = float(compactness)
    fraction = float(fraction)
    # Validate input parameters
    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")
    if not 0 <= compactness <= 1:
        raise ValueError("Compactness must be between 0 and 1.")

    # Initialize the array with zeros
    array = np.zeros(array_size, dtype=np.float32)
    # Calculate the target number of non-zero elements based on the desired fraction
    filled_volume = int(np.prod(array_size) * fraction)
    # Initialize the filled element count
    current_filled = 1

    # Choose a random starting point in the array
    start_point = tuple(random.randint(0, size - 1) for size in array_size)
    # Set the value of the starting point to a random value between 0 and 1
    array[start_point] = random.random()

    # Initialize a queue for breadth-first search and add the starting point
    queue = deque([start_point])

    # Helper function to get the neighboring points of a given point
    def get_neighbors(point):
        neighbors = []
        for dim, size in enumerate(array_size):
            if point[dim] > 0:
                neighbor = list(point)
                neighbor[dim] -= 1
                neighbors.append(tuple(neighbor))
            if point[dim] < size - 1:
                neighbor = list(point)
                neighbor[dim] += 1
                neighbors.append(tuple(neighbor))
        return neighbors

    # Breadth-first search to fill the non-zero elements
    while len(queue) > 0 and current_filled < filled_volume:
        current_point = queue.popleft()
        neighbors = get_neighbors(current_point)

        random.shuffle(neighbors)

        for neighbor in neighbors:
            # If the neighbor is 0 (unfilled) and the compactness condition is met, fill the neighbor with a random value
            if array[neighbor] == 0 and random.random() < compactness:
                array[neighbor] = random.random()
                queue.append(neighbor)
                current_filled += 1

                if current_filled >= filled_volume:
                    break

    # Smooth the array using a Gaussian filter with the specified smoothness
    smooth_array = gaussian_filter(array, sigma=smoothness)

    # Normalize the values to be between 0 and 1
    max_value = smooth_array.max()
    if max_value > 0:
        smooth_array /= max_value

    return smooth_array


from joblib import Parallel, delayed

def generate_multiple_smooth_functions(n_jobs, array_sizes, fractions, smoothnesses, compactnesses):
    return Parallel(n_jobs=n_jobs)(delayed(generate_smooth_function)(array_size, fraction, smoothness, compactness) 
                                   for array_size, fraction, smoothness, compactness in zip(array_sizes, fractions, smoothnesses, compactnesses))

# usage
#results = generate_multiple_smooth_functions(4, [(100,100,100), (100,100,100)], [0.3, 0.3], [10, 10], [0.4, 0.4])