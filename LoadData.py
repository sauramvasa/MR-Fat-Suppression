import os
import pydicom
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import CorruptImage
import random

plt.switch_backend('Qt5Agg')

def load_dicom_images(folder_path):
    # Get all DICOM files in the folder
 
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    images = np.zeros(len(file_paths)//2, dtype=np.ndarray)

    # Load each DICOM file into a numpy array
    for file_path in file_paths:
        if re.search('._', file_path):
            continue
        dataset = pydicom.dcmread(file_path)

        IN = dataset.InstanceNumber
        images[IN - 1] = dataset.pixel_array\
    
    # Stack the images into a 3D numpy array (height, width, depth)
    images = np.stack(images, axis=-1)

    return images

def load_multiple_series_of_dicom_images_and_data(folder_path, debug=False):

    #Get parts of folder name
    folder_name = folder_path.replace("/home/sauram/Documents/Research/MR-fatsup/Images/", '')
    parts_of_name = folder_name.split("-")

    #Determine whether the folder contains test data or training data and get body part and example number
    isTest = False
    body_part = parts_of_name[0]
    example_number = parts_of_name[1]
    if parts_of_name[0] == "test":
        isTest = True
        body_part = parts_of_name[1]
        example_number = parts_of_name[2]
    
    if debug:
        print(body_part + ", " + example_number + ", " + str(isTest))
    
    #Get the folders with DICOM files
    DICOM_folder_path = folder_path + "/DICOM"
    new_folder_path = DICOM_folder_path
    for i in range(3):
        unusable = os.listdir(new_folder_path)
        for path in range(len(unusable)):
            if re.search('._', unusable[path]):
                unusable[path] = 0
        unusable = [i for i in unusable if i != 0]
        new_folder_path1 = [os.path.join(new_folder_path, f) for f in unusable][0]
        if re.search('.DS', new_folder_path1):
            new_folder_path1 = [os.path.join(new_folder_path, f) for f in unusable][1]
        new_folder_path = new_folder_path1
    new_folder_paths_unusable = os.listdir(new_folder_path)
    for path in range(len(new_folder_paths_unusable)):
        if re.search('._', new_folder_paths_unusable[path]):
            new_folder_paths_unusable[path] = 0
    new_folder_paths_unusable = [i for i in new_folder_paths_unusable if i != 0]
    new_folder_paths = [os.path.join(new_folder_path, f) for f in new_folder_paths_unusable]
    for i in range(len(new_folder_paths)):
        if re.search('DS', new_folder_paths[i]):
            new_folder_paths.pop(i)
            break

    if debug:
        print(new_folder_paths)
        print(len(new_folder_paths))

    #Create array of zeros for images and information
    ImageSeriesArray = np.zeros(len(new_folder_paths), dtype=np.ndarray)
    InformationArray = np.zeros(len(new_folder_paths), dtype=np.ndarray)

    #Get the list of scores and contrast types from folder name
    if isTest:
        scores = parts_of_name[-len(new_folder_paths):]
        types = parts_of_name[-2 * len(new_folder_paths): -len(new_folder_paths)]
    else:
        scores = parts_of_name[-len(new_folder_paths)//2:]
        types = parts_of_name[-len(new_folder_paths): -len(new_folder_paths)//2]
    
    #Go into each folder
    index = 0
    for folder in new_folder_paths:

        #Load the data in the folder
        if re.search('._', folder):
            continue
        print(folder)
        series = load_dicom_images(folder)

        #Get one dicom file in the folder
        file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
        for path in file_paths:
            if not re.search('._', path):
                file_path = path
                break
        info_dataset = pydicom.dcmread(file_path)

        #Get flip angle, acquisition matrix, and MR acquisition type
        #flip_angle = info_dataset.FlipAngle
        #AQMatrix = info_dataset.AcquisitionMatrix
        MRAQType = info_dataset.MRAcquisitionType

        #Determine series number, series description, and whether it is a fat or water image
        SN = info_dataset.SeriesNumber
        BaseSN = SN
        isFat = False
        if SN % 100 == 0:
            isFat = True
            BaseSN = SN // 100
        SeriesDescription = info_dataset.SeriesDescription.split(" ")[1:]

        #Get the image contrast type
        ImageContrastType = ""
        if "post" in SeriesDescription:
            ImageContrastType = "post"
        elif "T1" in SeriesDescription and MRAQType == "2D":
            ImageContrastType = "T1"
        elif "T2" in SeriesDescription and MRAQType == "3D":
            ImageContrastType = "T2"
        elif "T2" in SeriesDescription and MRAQType == "2D":
            ImageContrastType = "T2"
        elif "MSDE" in SeriesDescription:
            ImageContrastType = "T2"
        elif "PD" in SeriesDescription:
            ImageContrastType = "PD"

        #Get the score using it's correspondence with the image contrast type in the folder name
        score = 0
        for i in range(len(types)):
            if ImageContrastType == types[i]:
                score = scores[i]
                break
        
        #Fill in the arrays
        ImageSeriesArray[index] = series
        InformationArray[index] = [ImageContrastType, "flip_angle", "AQMatrix", MRAQType, BaseSN, score, SeriesDescription, isFat]

        index += 1
    
    if debug:
        print(ImageSeriesArray)
        print(np.shape(ImageSeriesArray))
    
    #Create output list
    totally_perfect_array = []

    #Check if it is test or training data
    if not isTest:

        #Double iterate throught InformationArray
        for ind1 in range(len(InformationArray)):
            for ind2 in range(len(InformationArray)):

                #Make sure the indices are not the same
                if not ind1 == ind2:
                    
                    #Compare the series number to see if it is a fat water pair
                    if InformationArray[ind1][4] == InformationArray[ind2][4]:

                        #Check if it is totalswap
                        if  InformationArray[ind1][5] != "totalswap" and str(InformationArray[ind1][5]) != "0":

                            #Append a list of the information without isFat with a list of water image first, then fat image
                            if InformationArray[ind2][-1]:
                                totally_perfect_array.append([InformationArray[ind1][:-1], [ImageSeriesArray[ind1], ImageSeriesArray[ind2]]])
                            else:
                                totally_perfect_array.append([InformationArray[ind1][:-1], [ImageSeriesArray[ind2], ImageSeriesArray[ind1]]])
                        
                        #If it is totalswap, swap the order of the fat and water images
                        else:

                            # Change the score to a 5 and swap the fat and water
                            InformationArray[ind1][5] = '5'
                            if InformationArray[ind2][-1]:
                                totally_perfect_array.append([InformationArray[ind1][:-1], [ImageSeriesArray[ind2], ImageSeriesArray[ind1]]])
                            else:
                                totally_perfect_array.append([InformationArray[ind1][:-1], [ImageSeriesArray[ind1], ImageSeriesArray[ind2]]])
    
    #If it is a test, there are no pairs or totalswap shenanigans so it is simple
    else:
        for ind in range(len(InformationArray)):
            totally_perfect_array.append([InformationArray[ind], ImageSeriesArray[ind]])

    #Remove duplicates
    #Double iterate through the output array to see if any are the same
    for ind1 in range(len(totally_perfect_array)):
        for ind2 in range(len(totally_perfect_array)):
  
            #Make sure nothing is set to zero and indices are not the same
            if totally_perfect_array[ind1] != 0 and totally_perfect_array[ind2] != 0:
                if not ind1 == ind2:

                    #If the two have equal series numbers even after being connected that means they are duplicates
                    if totally_perfect_array[ind1][0][4] == totally_perfect_array[ind2][0][4]:

                        #Set one of them to zero
                        if ind1 > ind2:
                            totally_perfect_array[ind1] = 0
                        else:
                            totally_perfect_array[ind2] = 0
    
    #Get rid of all the zeros
    totally_perfect_array = [i for i in totally_perfect_array if i != 0]
    
    #Return the array containing each series and it's information
    return totally_perfect_array

def SimplifyData(data, plane='', Only2D=False):
    # Takes in one patient's loaded data

    new_data = data.copy()

    def get2DOnly(new_data):
        for i in range(len(new_data)):
            if new_data[i][0][3] == "3D":
                new_data[i] = 0
        
        new_data = [i for i in new_data if i != 0]
        return new_data
    
    if Only2D:    
        new_data = get2DOnly(new_data)
        if new_data == []:
            return False
    
    # Only use series of showing plane
    if not plane == '':
        for i in range(len(new_data)):
            if not new_data[i][0][6][0] == plane:
                print(plane)
                print(new_data[i][0][6][0])
                new_data[i] = 0

        new_data = [i for i in new_data if i != 0]

    if new_data == []:
        return False

    #Remove non 5 series
    data_with_only_5 = []
    for series in new_data:
        if str(series[0][5]) == "5":
            data_with_only_5.append(series)
        else:
            print("REMOVED BAD IMAGE PAIR")

    # Remove titles from data
    just_image_data = []
    for series_pair in data_with_only_5:
        just_image_data.append(series_pair[1])

    # Corrupt each series with corruption factors
    corrupted_data = []
    for image_pair3D in just_image_data:
        randfrac = random.randint(1, 2) * 0.1
        corrupted_data.append(CorruptImage.CorruptImageGenerator(image_pair3D[0], image_pair3D[1], fraction=randfrac))

    # Label each slice with its corruption factor, i.e. reshape
    # [[[ser1slice1,se1sl1factor], [ser1slice2, se1sl2factor],..., [se1sliceN, ser1sliceNfactor]], [se2]...]
    better_corrupted_data = []
    for corrupt_image in corrupted_data:
        image_factor_weight_stack = []
        for ind in range(len(corrupt_image[1])):
            image_factor_weight_stack.append([corrupt_image[0][0:len(corrupt_image[0]), 0:len(corrupt_image[0][0]), ind], corrupt_image[1][ind], corrupt_image[2][ind]])
            #if corrupt_image[1][ind] > 0.15: # 0.15 is the threshold for fat images
            #   image_factor_stack.append([corrupt_image[0][0:len(corrupt_image[0]), 0:len(corrupt_image[0][0]), ind], 1])
            #else:
            #   image_factor_stack.append([corrupt_image[0][0:len(corrupt_image[0]), 0:len(corrupt_image[0][0]), ind], 0])
        better_corrupted_data.append(image_factor_weight_stack)

    # Concatenates each series i.e. flattens to a list of pairs of 2D image/label
    even_better_corrupted_data = []
    for series_pair in better_corrupted_data:
        for corim in series_pair:
            even_better_corrupted_data.append(corim)

    # Remove slices with mostly air
    weights = []
    for image_group in even_better_corrupted_data:
        weights.append(image_group[2])

    max_weight = max(weights)
    min_weight = min(weights)
    cutoff = min_weight + (max_weight - min_weight) * 0.1
    bad_factor_removed = []
    for image_group in even_better_corrupted_data:
        if image_group[2] > cutoff:
            bad_factor_removed.append(image_group)
    
    return bad_factor_removed