print("Beginning execution of code")
import Utilities
import os
import LoadData
import FunctionGenerator
import CorruptImage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.profiler as profiler
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize, resize
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import glob as glob

print("Imported modules")

matplotlib.use("MacOSX")
plt.ion()


# Hyper-parameters
imsize = [128,128]
num_epochs = 30
batch_size = 250
num_augs = 5
learning_rate = 0.0001
print(imsize)


def create_dataset(image_list):
    for item in image_list:
        image = item[0]
        
        # Resize the image to the desired size
        resized_image = Utilities.resize_image(image, imsize)

        # Normalize the pixel values to range between 0 and 255
        normalized_image = (resized_image / np.amax(resized_image)) * 255.0

        # Transform the image to a tensor and normalize it
        intarr = normalized_image.astype(int)
        trans = transforms.Compose([transforms.ToTensor()])
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
        self.conv1_bn = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64*4*4, 2)
        #self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu5(x)
        x = self.dropout5(x)
        x = self.maxpool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        #output = self.softmax(x)      
        return x

def weighted_mse_loss(predictions, targets, weights=None):
    loss = (predictions - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def TrainCNN1(num_epochs, training_paths, num_augs, batch_size, target_loss=0, debug=False):
    
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
                print("for epoch  " + str(epoch+1) + "  on patient set " + str(index))
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
        
        torch.save(model.state_dict(), '/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth1')

def TrainCNN2(num_epochs, training_paths, num_augs, batch_size, target_loss=0, debug=False):
    
    # Turn training paths into smaller sublists to train on
    training_paths = [training_paths[i:i + 3] for i in range(0, len(training_paths), 3)]
    print(training_paths)

    # Create an instance of the CNN
    model = CNN()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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
            num_epochs = 20
            num_augs = 10
            training_paths = training_paths
        
        for epoch in range(num_epochs):
            print("On epoch " + str(epoch + 1))

            index = 0
            for patient_group in training_paths:
                index += 1
                print("On patient batch " + str(index))

                print("Loading")
                massive_train_dataset = []
                for patient in patient_group:

                    print(patient)
                    
                    # Load patient
                    loaded_patient = LoadData.load_multiple_series_of_dicom_images_and_data(patient)
                    
                    for aug in range(num_augs):

                        # Prepare the data for training
                        train_image_list = LoadData.SimplifyData(loaded_patient, Only2D=False)

                        #Display a random corrupted image if debug is on
                        if debug:
                            randind = random.randint(0, len(train_image_list) - 1)
                            plt.imshow(train_image_list[randind][0], cmap='gray')
                            plt.title(f'Image with Category {train_image_list[randind][1]:.4f}, Weight {train_image_list[randind][2]:.4f}, and Index {randind:.4f}')
                            fig4.canvas.draw()
                            fig4.canvas.flush_events()
                        
                        train_dataset = create_dataset(train_image_list)
                        
                        for slice in train_dataset:
                            massive_train_dataset.append(slice)
            
                train_dataloader = DataLoader(massive_train_dataset, batch_size=batch_size, shuffle=True)

                print("Training")
                running_loss = 0.0
                for images, labels, w in train_dataloader:
                    # Move the images and labels to the device
                    images = images.to(device)
                    labels = labels.to(device)
                    w = w.type('torch.FloatTensor').to(device)
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(torch.Tensor.float(images))
                    loss = criterion(outputs, torch.Tensor.long(labels))
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()

                loss_values.append(running_loss/len(train_dataloader))

                # Print and plot the average loss for this epoch
                print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(train_dataloader)}")
                plt.plot(loss_values)
                plt.yscale("log")
                fig1.canvas.draw()
                fig1.canvas.flush_events()

        torch.save(model.state_dict(), '/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth2')

def TrainCNN3(num_epochs, training_paths, num_augs, batch_size, target_loss=0, plane='', continuous=False, debug=False):
    
    # Turn training paths into smaller sublists to train on
    ppb = 2 # Patient Per Batch
    training_paths = [training_paths[i:i + ppb] for i in range(0, len(training_paths), ppb)]
    print(training_paths)

    # Create an instance of the CNN
    model = CNN()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the loss function and optimizer
    if continuous:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        accuracy_values = []

        # Faster training
        if debug:
            display = False
            num_epochs = 3
            num_augs = 2
            training_paths = training_paths[0:1]
        
        fig1, ax = plt.subplots(2)
        if display:
            fig4 = plt.figure()

        index = 0
        for patient_group in training_paths:
            index += 1
            print("On patient batch " + str(index))

            print("Loading")
            massive_train_dataset = []
            setlabels = []
            for patient in patient_group:
                print(patient)
                
                # Load patient
                loaded_patient = LoadData.load_multiple_series_of_dicom_images_and_data(patient)
                
                for aug in range(num_augs):
                    print("On augmentation " + str(aug))

                    # Prepare the data for training
                    train_image_list = LoadData.SimplifyData(loaded_patient, plane=plane, Only2D=False)

                    # Turn all factors to labels if not continuous
                    if not continuous:
                        actual_corrupted_values = []
                        for image_group in train_image_list:
                            actual_corrupted_values.append(image_group[1])
                            if image_group[1] <= 0.05:
                                image_group[1] = 0
                            #elif image_group[1] <= 0.1:
                                #image_group[1] = 1
                            #elif image_group[1] <= 0.30:
                            #    image_group[1] = 2
                            else:
                                image_group[1] = 1

                    # Display a random corrupted image if debug is on
                    if display:
                        randind = random.randint(0, len(train_image_list) - 1)
                        plt.imshow(train_image_list[randind][0], cmap='gray')
                        plt.title(f'Image with Category {train_image_list[randind][1]:.4f}, Weight {train_image_list[randind][2]:.4f}, and Index {randind:.4f}')
                        fig4.canvas.draw()
                        fig4.canvas.flush_events()
                    
                    train_dataset = create_dataset(train_image_list)
                    
                    for slice in train_dataset:
                        massive_train_dataset.append(slice)
                        setlabels.append(slice[1])
            
            print(setlabels)
            train_dataloader = DataLoader(massive_train_dataset, batch_size=batch_size, shuffle=True)

            for epoch in range(num_epochs):
                print("Training on epoch " + str(epoch + 1))
                running_loss = 0.0
                correct = 0
                dataloader_counter = 0
                for images, labels, w in train_dataloader:
                    # Move the images and labels to the device
                    images = images.to(device)
                    labels = labels.to(device)
                    w = w.type('torch.FloatTensor').to(device)
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(torch.Tensor.float(images))
                    if continuous:
                        loss = criterion(outputs, torch.Tensor.float(labels).unsqueeze(1))
                    else:
                        _, predicted = torch.max(outputs.data, 1)
                        loss = criterion(outputs, torch.Tensor.long(labels))

                        # Get accuracy
                        for i in range(len(predicted.tolist())):
                            dataloader_counter += 1
                            if labels[i] == predicted[i]:
                                correct += 1
                    
                    # Backward pass and optimization
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()

                loss_values.append(running_loss/len(train_dataloader))
                accuracy_values.append(correct/dataloader_counter)

                # Print and plot the average loss for this epoch
                print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss/len(train_dataloader)}, Accuracy: {correct/dataloader_counter}")
                ax[0].plot(loss_values)
                ax[0].set_yscale("log")
                ax[0].set_title("Training Loss")
                ax[1].plot(accuracy_values)
                ax[1].set_title("Training Accuracy")
                fig1.tight_layout(pad=2.0)
                fig1.canvas.draw()
                fig1.canvas.flush_events()

        torch.save(model.state_dict(), '/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth4')

images_shown = []
titles_shown = []
def TrainCNN4(num_epochs, training_paths, num_augs, batch_size, target_loss=0, debug=False):
    
    # Create an instance of the CNN
    model = CNN()

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Move the loss function to the device
    criterion.to(device)

    # Check if CNN already exists
    #if os.path.exists('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth'):
        #pass

    # The CNN does not exist

    # Set the model in training mode move the model to the device
    model.to(device)
    model.train()

    loss_values = []

    flagDisplay = False

    Only2D = debug
    if debug:
        training_paths = training_paths
        num_epochs = 50
        num_augs = 30
    
    index=1
        
    #if debug:
        #training_paths = training_paths[0:2]

    massive_train_dataset = []
    
    for patient in training_paths:
        print("On patient #" + str(index))
        index += 1

        # Load patient
        loaded_patient = LoadData.load_multiple_series_of_dicom_images_and_data(patient)
        
        for aug in range(num_augs):

            # Prepare the data for training
            train_image_list = LoadData.SimplifyData(loaded_patient, Only2D=False)

            if not train_image_list:
                print("3D image")
                break

            if debug:
                randind = random.randint(0, len(train_image_list) - 1)
                plt.imshow(train_image_list[randind][0], cmap='gray')
                plt.title(f'Image with Category {train_image_list[randind][1]:.4f}, Weight {train_image_list[randind][2]:.4f}, and Index {randind:.4f}')
                #fig4.canvas.draw()
                #fig4.canvas.flush_events()

            if aug == 0:
                images_shown.append(train_image_list[len(train_image_list)//2][0])
                titles_shown.append(train_image_list[len(train_image_list)//2][1])
                if flagDisplay:
                    Utilities.display_dynamic_3d_array(images_shown, titles_shown)
                    flagDisplay = False
            
            train_dataset = create_dataset(train_image_list)
            
            for slice in train_dataset:
                massive_train_dataset.append(slice)

    print("Reached dataloader")
    train_dataloader = DataLoader(massive_train_dataset, batch_size=batch_size, shuffle=True)

    print("Finished Loading Data, Started Training")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels, w in train_dataloader:
            # Move the images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            w = w.type('torch.FloatTensor').to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(torch.Tensor.float(images))
            loss = criterion(outputs, torch.Tensor.long(labels))
            
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
    
    print("Loss Values:    " + str(loss_values))

    torch.save(model.state_dict(), '/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth4')

storeTests=[]
def TestCNN(test_paths, batch_size, continuous=False, plane='', debug=False):

    # Assume the model has been trained and saved the regression model as 'model.pth'
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth3'))
    model.eval()

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("mps")
    model.to(device)

    if continuous:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if debug:
        #test_paths = ["/Users/sasv/Documents/Research/MR-fatsup/Images/ankle-3-PD-5"]
        test_paths = test_paths
    
    for patient in test_paths:

        # Load patient
        LoadedTestData = LoadData.load_multiple_series_of_dicom_images_and_data(patient)

        # Prepare the data for testing
        test_image_list = LoadData.SimplifyData(LoadedTestData, plane=plane)
        storeTests.append(test_image_list)

        # Turn all factors to labels if not continuous
        if not continuous:
            actual_corrupted_values = []
            for image_group in test_image_list:
                actual_corrupted_values.append(image_group[1])
                if image_group[1] <= 0.05:
                    image_group[1] = 0
                #elif image_group[1] <= 0.1:
                    #image_group[1] = 1
                #elif image_group[1] <= 0.30:
                #    image_group[1] = 2
                else:
                    image_group[1] = 1

        dataloader_ready_testdataset = create_dataset(test_image_list)

        # Create a DataLoader for the test dataset 
        test_dataloader = DataLoader(dataloader_ready_testdataset, batch_size=batch_size, shuffle=False)

        # Lists to store the actual and predicted values
        actual_values = []
        predicted_values = []

        plot_actual_values = []
        plot_predicted_values = []

        print("Started testing")
        # Iterate over the test dataset and make predictions
        with torch.no_grad():
            for images, labels, w in test_dataloader:
                # Move the images to the device
                images = images.to(device)
                
                # Forward pass and get the predicted values
                outputs = model(torch.Tensor.float(images))
                for output in outputs:
                    predicted_values.append(output.tolist())
                    if not continuous:
                        plot_predicted_values.append(torch.argmax(output))
                    else:
                        plot_predicted_values.append(output.tolist())
                for label in labels:
                    actual_values.append(label.item())
                    plot_actual_values.append(label.item())
                #predicted_value = outputs.item()
                
                # Append the actual and predicted values to the respective lists
                #actual_values.append(labels.item())
                #predicted_values.append(predicted_value)
        print(torch.tensor(actual_values))
        print(torch.tensor(predicted_values))

        # Calculate the test error
        test_error = criterion(torch.tensor(predicted_values), torch.tensor(actual_values))

        # Plot the actual and predicted values
        '''
        fig2 = plt.figure()
        plt.plot(plot_actual_values, label='Actual Labels')
        plt.plot(plot_predicted_values, label='Predicted Labels')
        plt.plot(actual_corrupted_values, label='Actual Factors')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
        fig2.legend()
        fig2.show()
        '''

        if not continuous:
            fig2, ax2 = plt.subplots()
            ax2.plot(plot_actual_values, label='Actual Labels')
            ax2.plot(plot_predicted_values, label='Predicted Labels')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('0 or 1 Value')
            ax2.set_title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
            axNew = ax2.twinx()
            axNew.plot(actual_corrupted_values, color='green', label='Actual Factors')
            axNew.set_ylabel('Actual Value')
            fig2.legend()
            fig2.show()

            col_labels = ['Predicted 0', 'Predicted 1']
            row_labels = ['Actual 0', 'Actual 1']
            confMatrix = [[0,0],[0,0]]
            for ind in range(len(actual_values)):
                confMatrix[plot_actual_values[ind]][plot_predicted_values[ind].item()] += 1

            fig3 = plt.figure()
            plt.scatter(plot_actual_values, plot_predicted_values)
            plt.xlabel('Actual Corruption Factor')
            plt.ylabel('Predicted Corruption Factor')
            plt.title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
            plt.axline((0, 0), slope=1)
            plt.table(cellText=confMatrix, colWidths=[0.1] * 4, rowLabels=row_labels, colLabels=col_labels, loc='upper right')
            fig3.show()
        else:
            fig2, ax2 = plt.subplots()
            ax2.plot(plot_actual_values, label='Actual Factors')
            ax2.plot(plot_predicted_values, label='Predicted Factors')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Factor')
            ax2.set_title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
            fig2.legend()
            fig2.show()

            fig3 = plt.figure()
            plt.scatter(plot_actual_values, plot_predicted_values)
            plt.xlabel('Actual Corruption Factor')
            plt.ylabel('Predicted Corruption Factor')
            plt.title(f'Actual vs Predicted Values\nTest Error: {test_error:.4f}')
            plt.axline((0, 0), slope=1)
            fig3.show()

def VisualizeFeatureMaps():
    # Assume the model has been trained and saved the regression model as 'model.pth'
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth3'))

    #save the conv layer weights in this list
    model_weights = []
    #save the conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print(conv_layers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load patient
    LoadedImage = LoadData.load_multiple_series_of_dicom_images_and_data("/Users/sasv/Documents/Research/MR-fatsup/Images/thigh-2-T2-T2-post-4-4-0")

    fig5 = plt.figure()

    # Prepare the data for testing
    test_image_list = LoadData.SimplifyData(LoadedImage)
    dataloader_ready_testdataset = create_dataset(test_image_list)
    image = dataloader_ready_testdataset[100][0]
    plt.imshow(test_image_list[100][0].reshape(128,128), cmap='gray')
    plt.title(f'Image with Category {test_image_list[100][1]:.4f}, Weight {test_image_list[100][2]:.4f}, and Index {100:.4f}')
    fig5.canvas.draw()
    fig5.canvas.flush_events()

    results = [conv_layers[0](torch.Tensor.float(image))]
    for layer in conv_layers[1:]:
        results.append(layer(results[-1]))
    print(len(results))

    #print feature_maps
    for feature_map in results:
        print(feature_map.shape)

    for num_layer in range(len(results)):
        plt.figure(figsize=(50,10))
        layer_viz = results[num_layer].squeeze()
        print("Layer", num_layer+1)
        for i, f in enumerate(layer_viz):
            plt.subplot(8, 8, i+1)
            plt.imshow(f.detach().cpu().numpy())
            plt.axis("off")
        plt.show()

def VisualizeFirstLayerFilters(): 
    # instantiate model
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth3'))

    # get the kernels from the first layer
    # as per the name of the layer
    kernels = model.conv1.weight.detach().clone()

    #check size for sanity check
    print(kernels.size())

    # normalize to (0,1) range so that matplotlib
    # can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    filter_img = torchvision.utils.make_grid(kernels, nrow = 12)
    # change ordering since matplotlib requires images to 
    # be (H, W, C)
    plt.imshow(filter_img.permute(1, 2, 0))



def display_class_activation_map1():

    # instantiate model
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth3'))

    # Set the model to evaluation mode
    model.eval()

    # Load patient
    LoadedImage = LoadData.load_multiple_series_of_dicom_images_and_data("/Users/sasv/Documents/Research/MR-fatsup/Images/thigh-2-T2-T2-post-4-4-0")

    # Prepare the data for testing
    test_image_list = LoadData.SimplifyData(LoadedImage)
    dataloader_ready_testdataset = create_dataset(test_image_list)
    image_tensor = dataloader_ready_testdataset[100][0]

    # Forward pass through the model to get the feature maps
    #feature_maps = model.feature_extraction(image_tensor)
    #save the conv layer weights in this list
    model_weights = []
    #save the conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print(conv_layers)

    feature_maps = [conv_layers[0](torch.Tensor.float(image_tensor))]
    for layer in conv_layers[1:]:
        feature_maps.append(layer(feature_maps[-1]))
    print(len(feature_maps))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Get the class predictions from the model
    class_predictions = model(torch.Tensor.float(image_tensor).unsqueeze(0))
    print(class_predictions)

    # Calculate the gradients of the class score with respect to the feature maps
    class_predictions.sum().backward()

    # Get the gradients from the last convolutional layer
    gradients = model.conv5.weight.grad

    # Compute the weights as the mean of the gradients
    weights = torch.mean(gradients.squeeze(), dim=(1, 2))

    # Multiply the weights by the feature maps
    print(weights.unsqueeze(1, ).size())
    print(torch.Tensor(feature_maps[4]).squeeze().size())
    print(weights.unsqueeze(1) * torch.Tensor(feature_maps[4]).squeeze())
    cam = torch.sum(weights.unsqueeze(1) * torch.Tensor(feature_maps[4]).squeeze(), dim=0)

    # Normalize the CAM
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)

    # Convert the CAM to a NumPy array
    cam = cam.detach().numpy()

    # Resize the CAM to match the original image size
    cam = np.array(image_tensor.fromarray(cam).resize(image_tensor.shape))

    # Plot the original image and the CAM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image_tensor, cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original Image')
    ax2.imshow(image_tensor, cmap='gray')
    ax2.imshow(cam, cmap='jet', alpha=0.5)
    ax2.axis('off')
    ax2.set_title('Class Activation Map')

    # Show the plot
    plt.show()

def guided_backpropagation(cnn_model, image_tensor):
    # Set the model to evaluation mode
    cnn_model.eval()

    # Convert the image to a PyTorch tensor
    image_tensor.requires_grad_()

    # Forward pass through the model to get the output
    output = cnn_model(torch.Tensor.float(image_tensor).unsqueeze(0))

    # Get the predicted class index
    _, predicted_class_index = torch.max(output, dim=1)
    predicted_class_index = predicted_class_index.item()

    # Zero out all the gradients
    cnn_model.zero_grad()

    # Compute gradients using guided backpropagation
    guided_gradients = torch.autograd.grad(output[0, predicted_class_index], image_tensor)[0]

    # Convert the gradients to numpy array
    guided_gradients = guided_gradients.detach().numpy()

    return guided_gradients[0, 0]  # Return the gradients as a 2D numpy array


def display_class_activation_map():

    # instantiate model
    model = CNN()
    model.load_state_dict(torch.load('/Users/sasv/Documents/Research/MR-fatsup/Models/model.pth3'))

    # Set the model to evaluation mode
    model.eval()

    # Load patient
    LoadedImage = LoadData.load_multiple_series_of_dicom_images_and_data("/Users/sasv/Documents/Research/MR-fatsup/Images/thigh-2-T2-T2-post-4-4-0")

    # Prepare the data for testing
    test_image_list = LoadData.SimplifyData(LoadedImage, plane='Ax')
    dataloader_ready_testdataset = create_dataset(test_image_list)
    image_tensor = torch.Tensor.float(dataloader_ready_testdataset[100][0])
    #image_tensor.requires_grad_(True)

    # Get the feature maps from the target layer
    # Create empty list to store the feature maps
    feature_maps = []

    # Define the hook function to store the feature maps
    def hook_fn(module, input, output):
        feature_maps.append(output.detach())

    # Register the hook to the target layer
    hook_handle = model.conv5.register_forward_hook(hook_fn)

    # Forward pass through the model to get the final output
    outputs = model(torch.Tensor.float(image_tensor).unsqueeze(0))

    # Get the predicted class index
    _, predicted_class_index = torch.max(outputs, dim=1)
    predicted_class_index = predicted_class_index.item()

    # Retrieve the feature maps from the list
    extracted_feature_maps = feature_maps[0]

    # Remove the hook to avoid affecting subsequent forward passes
    hook_handle.remove()

    #extracted_feature_maps.requires_grad_(True)

    # Compute the guided gradients using guided backpropagation
    guided_gradients = guided_backpropagation(model, image_tensor)

    # Compute the weights as the global average pooling of the guided gradients
    weights = torch.mean(torch.from_numpy(guided_gradients), dim=0, keepdim=True)

    # Multiply the weights by the feature maps
    cam = torch.sum(weights.unsqueeze(1) * extracted_feature_maps.squeeze(), dim=0)

    # Normalize the CAM
    cam = F.relu(cam)
    cam = cam - torch.min(cam)
    cam = cam / torch.max(cam)

    # Convert the CAM to a NumPy array
    cam = cam.detach().numpy()

    # Resize the CAM to match the original image size
    cam = np.array(Image.fromarray(cam).resize((image_tensor.detach().numpy().shape[2], image_tensor.detach().numpy().shape[1])))
    print(cam.shape)
    print(predicted_class_index)
    print(dataloader_ready_testdataset[100][1])

    # Plot the original image and the CAM
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image_tensor.squeeze(0).detach().numpy(), cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original Image')
    ax2.imshow(image_tensor.squeeze(0).detach().numpy(), cmap='gray')
    ax2.imshow(cam, cmap='jet', alpha=0.5)
    ax2.axis('off')
    ax2.set_title('Class Activation Map')

    # Show the plot
    plt.show()



image_folder_paths = Utilities.get_image_paths(bodyType='thigh')
training_paths, test_paths, validation_paths = Utilities.separate_data(image_folder_paths, 0.60, 0.20, 0.20)

print("Training folders")
print(training_paths)
print("Testing folders")
print(test_paths)
print("Training folders length")
print(len(training_paths))
print("Testing folders length")
print(len(test_paths))

with profiler.profile(activities=[profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
    with profiler.record_function("model_inference"):
        TrainCNN3(num_epochs, training_paths, num_augs, batch_size, plane='Ax', continuous=False, debug=True)
        #TestCNN(test_paths, batch_size, continuous=False, plane='Ax', debug=True)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

#VisualizeFeatureMaps()
#display_class_activation_map()