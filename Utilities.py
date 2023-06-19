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
from matplotlib.widgets import Slider
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from collections import deque
import plotly.graph_objs as go

def get_image_paths():
    file_paths_in_image_folder = [os.path.join("/Users/sasv/Documents/Research/MR-fatsup/Images", f) for f in os.listdir("/Users/sasv/Documents/Research/MR-fatsup/Images")]
    for i in range(len(file_paths_in_image_folder)):
            if re.search('DS', file_paths_in_image_folder[i]):
                file_paths_in_image_folder.pop(i)
                break
    return file_paths_in_image_folder

def separate_data(paths, trainingpercentage, testpercentage, valpercentage):
    ''' Function input is a list of folders, percentage of folders that should be training, test, and validation'''

    training_list = paths[:int(len(paths)*trainingpercentage)]
    test_list = paths[int(len(paths)*trainingpercentage):int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)]
    validation_list = paths[int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage):int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage)]

    # Make sure no paths were left out
    if int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage) < len(paths):
        training_list.append(paths[int(len(paths)*trainingpercentage)+int(len(paths)*testpercentage)+int(len(paths)*valpercentage):len(paths)])

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