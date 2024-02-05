import numpy as np
from PIL import Image
import sys
import os

def mse_(image_1, image_2, mask=None):
    imageA = np.asarray(image_1)
    imageB = np.asarray(image_2)
    if mask is not None:
        imageA = imageA * mask
        imageB = imageB * mask
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    if mask is not None:
        err /= float(mask.sum())
    else:
        err /= float(imageA.shape[0] * imageA.shape[1])

    return err
