import Utilities
import LoadData
import FunctionGenerator
import math
import numpy as np
from scipy.ndimage import zoom

def CorruptImageGenerator(NonFatImages, FatImages, fraction=.3, smoothness=10, compactness=.4):
    NumImages = len(FatImages[0][0])
    width = len(FatImages[0])
    length = len(FatImages)

    ScaledFatImages = np.zeros([length, width, NumImages])

    zoomFactor = 4
    smoothFunc = zoom(FunctionGenerator.generate_smooth_function((math.ceil(NumImages/4), int(length/4), int(width/4)), fraction, smoothness, compactness), zoomFactor, order=0)

    # Scale eachs fat slice
    for ImageIndex in range(NumImages):
        ScaledFatImage = FatImages[0:length, 0:width, ImageIndex] * smoothFunc[ImageIndex]
        ScaledFatImages[0:length, 0:width, ImageIndex] = ScaledFatImage

    CorruptedImages = ScaledFatImages + NonFatImages

    # Get weights and corruption factors
    weights = []
    FactorOfCorruptionArr = np.zeros(NumImages)
    for ImageIndex in range(NumImages):
        ScaledFatImageSum = np.sum(np.sum(ScaledFatImages[0:length, 0:width, ImageIndex]))
        NonFatImageSum = np.sum(np.sum(NonFatImages[0:length, 0:width, ImageIndex]))
        weights.append(np.mean(NonFatImages[0:length, 0:width, ImageIndex]))
        FatImageSum = np.sum(np.sum(FatImages[0:length, 0:width, ImageIndex]))
        FactorOfCorruptionArr[ImageIndex] = ScaledFatImageSum / (NonFatImageSum + FatImageSum)
    
    return [CorruptedImages, FactorOfCorruptionArr, weights]