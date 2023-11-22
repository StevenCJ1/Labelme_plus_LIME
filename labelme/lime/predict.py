from pprint import pprint
import os
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.color
import skimage.segmentation
import matplotlib
matplotlib.use('TkAgg')
from PyQt5.QtGui import QImage



def convertQImageToMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''

    incomingImage = incomingImage.convertToFormat(4)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  #  Copies the data
    return arr

def explained_module_predict(img, num_top_guess, model, shape):
    explained_module = model
    top_guesses = num_top_guess  # Integer, how many top-guesses to return.


    image_rgba = convertQImageToMat(img)

    if np.shape(image_rgba)[2] != 3:
        image = skimage.color.rgba2rgb(image_rgba)
    else:
        image = image_rgba

    image = skimage.transform.resize(image, (shape, shape))
    image = (image - 0.5) * 2  # Inception pre-processing

    preds = explained_module.predict(image[np.newaxis, :, :, :], verbose = 0)
    top_pred = decode_predictions(preds, top=top_guesses)[0]
    
    return top_pred



if __name__ == "__main__":
    all_preds = []
    image = QImage()
    image.load("image_file/two_dog.JPEG")  # Replace with the actual path to your image file

    # Convert sip.voidptr to a NumPy array
    top_pred_list = explained_module_predict(image, 10)
    for i, pred_tuple in enumerate(top_pred_list):
        pred_name = pred_tuple[1]
        pred_prob = str(pred_tuple[2])
        pred = str(i) + "." + pred_name + " " +pred_prob
        all_preds.append(pred)
        new_info = "\n".join(all_preds)

    print(new_info)
