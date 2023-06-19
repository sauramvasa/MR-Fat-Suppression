print("Beginning execution of code")
import Utilities
import os
import LoadData
import FunctionGenerator
import CorruptImage
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
print("Imported modules")

plt.ion()
fig1 = plt.figure()

# Hyper-parameters
imsize = [128,128]
num_epochs = 30
batch_size = 10
num_augs = 1
learning_rate = 0.001
trans = transforms.Compose([transforms.ToTensor()])
print(imsize)


def create_dataset(image_list):
    for item in image_list:
        image = item[0]
        class_label = item[1]
        
        # Resize the image to the desired size
        resized_image = Utilities.resize_image(image, imsize)
        
        # Normalize the pixel values to range between 0 and 255
        normalized_image = (resized_image / 32768) * 255.0

        # Transform the image to a tensor and normalize it
        intarr = normalized_image.astype(int)
        transedintarr = trans(intarr)
        
        # Append the normalized image and its label to the lists
        item[0] =  transedintarr

    return image_list
        
# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

# Define your CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64*16*16, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

'''
# Create an instance of the CNN
model = CNN()

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
image_folder_paths = Utilities.get_image_paths()
folder_path = "/Users/sasv/Documents/Research/MR-fatsup/Images/ankle-4-PD-PD-5-5"

LoadedTrainingData = LoadData.load_multiple_series_of_dicom_images_and_data(folder_path)
print("Loaded training data.")

train_image_list = LoadData.SimplifyData(LoadedTrainingData)
num_images = len(train_image_list)
print(num_images)

# Create a DataLoader for the training set
dataloader_ready_dataset = create_dataset(train_image_list)  # Resizes and normalizes images to 128x128
train_dataloader = DataLoader(dataset=dataloader_ready_dataset, batch_size=batch_size, shuffle=True)
'''
'''
examples = iter(train_dataloader)
example_data, example_targets = examples.next()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show
'''
'''
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move the model and loss function to the device
model.to(device)
criterion.to(device)
'''
def TrainCNN(num_epochs, training_paths, num_augs, batch_size, target_loss=0, debug=False):
    
    # Create an instance of the CNN
    model = CNN()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move the loss function to the device
    criterion.to(device)

    # Check if CNN already exists
    #if os.path.exists('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth'):
        #pass

    # The CNN does not exist
    if debug:

        # Set the model in training mode move the model to the device
        model.to(device)
        model.train()

        loss_values = []

        # Faster training
        if debug:
            num_epochs = 5
            training_paths = training_paths
        
        for epoch in range(num_epochs):
            index=1
            for patient in training_paths:
                print("for epoch  " + str(epoch+1) + "  on patient " + str(index))
                index += 1

                # Load patient
                loaded_patient = LoadData.load_multiple_series_of_dicom_images_and_data(patient)
                for aug in range(num_augs):

                    # Prepare the data for training
                    train_image_list = LoadData.SimplifyData(loaded_patient)
                    train_dataset = create_dataset(train_image_list) 
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    running_loss = 0.0
                    for images, labels in train_dataloader:
                        # Move the images and labels to the device
                        images = images.to(device)
                        labels = labels.to(device)
                        # Zero the gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        outputs = model(torch.Tensor.float(images))
                        loss = criterion(outputs, torch.Tensor.float(labels))
                        
                        # Backward pass and optimization
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                    
                    loss_values.append(running_loss/len(train_dataloader))
                    # Print the average loss for this epoch
                    print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(train_dataloader)}")
                    plt.plot(loss_values)
                    plt.yscale("log")
                    #plt.ylim([0, 1])
                    fig1.canvas.draw()
                    fig1.canvas.flush_events()
        
        torch.save(model.state_dict(), '/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth')



def TestCNN(test_paths, batch_size, debug=False):

    # Assume the model has been trained and saved the regression model as 'model.pth'
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth'))
    model.eval()

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if debug:
        #test_paths = ["/Users/sasv/Documents/Research/MR-fatsup/Images/ankle-3-PD-5"]
        test_paths = test_paths
    
    for patient in test_paths:

        # Load patient
        LoadedTestData = LoadData.load_multiple_series_of_dicom_images_and_data(patient)

        # Prepare the data for testing
        test_image_list = LoadData.SimplifyData(LoadedTestData)
        dataloader_ready_testdataset = create_dataset(test_image_list)

        # Create a DataLoader for the test dataset
        test_dataloader = DataLoader(dataloader_ready_testdataset, batch_size=batch_size, shuffle=False)

        # Lists to store the actual and predicted values
        actual_values = []
        predicted_values = []

        print("Started testing")
        # Iterate over the test dataset and make predictions
        with torch.no_grad():
            for images, labels in test_dataloader:
                # Move the images to the device
                images = images.to(device)
                
                # Forward pass and get the predicted values
                outputs = model(torch.Tensor.float(images))
                print(outputs.shape)
                for output in outputs:
                    predicted_values.append(output.item())
                for label in labels:
                    actual_values.append(label.item())
                #predicted_value = outputs.item()
                
                # Append the actual and predicted values to the respective lists
                #actual_values.append(labels.item())
                #predicted_values.append(predicted_value)
        print(actual_values)
        print(predicted_values)

        # Calculate the test error
        mse = nn.MSELoss()
        test_error = mse(torch.tensor(predicted_values), torch.tensor(actual_values)).item()

        # Plot the actual and predicted values
        fig2 = plt.figure()
        plt.plot(actual_values, label='Actual')
        plt.plot(predicted_values, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
        fig2.legend()
        fig2.show()

        fig3 = plt.figure()
        plt.scatter(actual_values, predicted_values)
        plt.xlabel('Actual Corruption Factor')
        plt.ylabel('Predicted Corruption Factor')
        plt.title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
        plt.axline((0, 0), slope=1) 
        fig3.show()

image_folder_paths = Utilities.get_image_paths()
training_paths, test_paths, validation_paths = Utilities.separate_data(image_folder_paths, 0.60, 0.20, 0.20)
print("Training folders")
print(training_paths)
print("Testing folders")
print(test_paths)
print("Training folders length")
print(len(training_paths))
print("Testing folders length")
print(len(test_paths))
TrainCNN(num_epochs, training_paths, num_augs, batch_size, debug=True)        
TestCNN(test_paths, batch_size, debug=True)    