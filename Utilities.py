import os
import LoadData
import FunctionGenerator
import CorruptImage
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pydicom
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from collections import deque
import plotly.graph_objs as go

import matplotlib

def get_image_paths(bodyType=''):
    file_paths_in_image_folder = [os.path.join("/Users/sasv/Documents/Research/MR-fatsup/Images", f) for f in os.listdir("/Users/sasv/Documents/Research/MR-fatsup/Images")]
    for i in range(len(file_paths_in_image_folder)):
        if re.search('DS', file_paths_in_image_folder[i]):
            file_paths_in_image_folder.pop(i)
            break
    if not bodyType == '':
        for i in range(len(file_paths_in_image_folder)):
            if not re.search(bodyType, file_paths_in_image_folder[i]):
                file_paths_in_image_folder[i] = 0
    file_paths_in_image_folder = [i for i in file_paths_in_image_folder if i != 0]
    return file_paths_in_image_folder

def separate_data(paths, trainingpercentage, testpercentage, valpercentage):
    ''' Function input is a list of folders, percentage of folders that should be training, test, and validation'''

    training_list = paths[:int(len(paths)*trainingpercentage)]
    test_list = paths[int(len(paths)*trainingpercentage):int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)]
    validation_list = paths[int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage):int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage)]

    # Make sure no paths were left out
    if int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage) < len(paths):
        for path in paths[int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage):len(paths)]:
            training_list.append(path)

    return training_list, test_list, validation_list

def display_dicom_images(images):
    fig, ax = plt.subplots()
    ax.imshow(images[:, :, 0], cmap='gray')
    ax.set_title('Image 0')
    plt.show(block=False)
    
    def update_image(index):
        ax.imshow(images[:, :, index], cmap='gray')
        ax.set_title(f'Image {index}')
        fig.canvas.draw()
    
    # Create a slider widget to scroll through the images
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.04])
    slider = Slider(ax_slider, 'Image', valmin=0, valmax=images.shape[-1] - 1, valinit=0, valstep=1)
    slider.on_changed(update_image)
    
    plt.show()
    
    return slider

def plot_numpy_arrays_on_slider(numpy_array, ax_slider, slider):
    # Get the current index and maximum value of the slider
    current_index = int(slider.val)
    max_index = int(slider.valmax)

    # Increment the maximum value of the slider
    slider.valmax += 1

    # Update the slider properties
    slider.ax.set_xlim(slider.valmin, slider.valmax)
    slider.ax.set_xticks(np.arange(slider.valmin, slider.valmax + 1, 1))

    # Add the new numpy array to the slider
    ax_slider.sliderdict[slider.cid][0].append(numpy_array)

    # Update the plot based on the new maximum value of the slider
    if current_index == max_index:
        slider.set_val(max_index + 1)

    plt.draw()
    plt.show()


def plot_3d_isosurface(data, threshold):
    # Extract dimensions of the data array
    n_x, n_y, n_z = data.shape

    # Generate integer grids
    X, Y, Z = np.mgrid[:n_x, :n_y, :n_z]

    # Create the isosurface plot
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=data.flatten(),
        isomin=threshold,
        isomax=data.max(),
        opacity=0.6,
        surface=dict(count=2, fill=0.8, pattern='odd'),
        colorscale='Viridis'
    ))

    # Customize the plot
    fig.update_layout(
        title='3D Isosurface Plot',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    # Show the plot
    fig.show()

def resize_image(image, size):
    # Resize the image using your preferred method (e.g., using a library like PIL)
    # Implement the logic to resize the image based on the given size
    # Return the resized image
    
    # Example using PIL:
    from PIL import Image
    image = Image.fromarray(image)
    resized_image = image.resize(size)
    resized_image = np.array(resized_image)
    
    return resized_image

def display_dynamic_3d_array(array, titles):
    # Initialize the figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Create the initial plot with the first 2D array from the 3D array
    current_index = 0
    current_array = array[current_index]
    img = ax.imshow(current_array, cmap='gray')
    ax.set_title(titles[current_index])

    # Configure the slider
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Index', 0, len(array) - 1, valinit=current_index, valstep=1)

    # Update the plot based on the slider value
    def update(val):
        nonlocal current_index, current_array
        current_index = int(val)
        current_array = array[current_index]
        img.set_data(current_array)
        ax.set_title(titles[current_index])
        fig.canvas.draw_idle()

    # Connect the slider update function to the slider value change event
    slider.on_changed(update)

    # Function to check for modifications in the array
    def check_modifications(new_array):
        nonlocal array
        if not np.array_equal(array, new_array):
            array = new_array
            slider.set_val(0)  # Reset the slider to the first index
            update(0)  # Update the plot

    # Show the plot
    plt.show()