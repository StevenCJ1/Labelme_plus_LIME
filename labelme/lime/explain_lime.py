from pprint import pprint
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
import skimage.color
import skimage.segmentation
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings
import cv2
from labelme.lime import test




def perturb_image(img, perturbation, segments):
    """
    Generates images and predictions in the neighborhood of this image.

    input: image: 3d numpy array, the image
           perturbation: the marks of numpy array [0, 1]
           segments: segmentation of the image

    return: perturbed_image
    """
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image


def mix_segment(lbl, label_names, superpixels):
    """
    Mix the segmentation from SLIC and labelme.exe(users)

    input: json_file: the interactive segmentation
           superpixels: segmentation from SLIC

    return: accumulateSPs: mixed segmentation
            interactive_names: the name that user mark in labelme.exe
    """
    accumulateSPs = superpixels
    interactive_SPs = lbl
    interactive_names = label_names
    # We have to use the reserve array seq[::-1]
    # seq[start:stop:step] => a slice from start to stop, stepping step each time.
    for i in np.unique(interactive_SPs)[1:][::-1]:
        separationSPs = np.where(interactive_SPs == i, i, 0)
        separationSPs = skimage.transform.resize(separationSPs, (299, 299))
        separationSPs = np.where(separationSPs == 0, 1, 0)
        accumulateSPs = (accumulateSPs + 1) * separationSPs  # plus 1 to ensure dont cover originally segmentation 0

    num_SPs = np.unique(accumulateSPs).shape[0]
    i = 0
    while i < num_SPs:
        if np.where(accumulateSPs == i)[0].size == 0:
            indices = np.where(accumulateSPs > i)
            dist = np.min(accumulateSPs[indices]) - i
            accumulateSPs = np.where(accumulateSPs > i, accumulateSPs - dist, accumulateSPs)
        else:
            i = i + 1
    return accumulateSPs, interactive_names


def get_num_segmentSPs(lbl, label_names):
    '''
    Set the number of superpixels that SLIC should segment
    The size of interactive superpixels should same as SLIC superpixels.
    During calculate the size of interactive superpixel to get the size of SLIC superpixels,
    and then get the total number of superpixels from SLIC.

    input: json_file: the interactive segmentation

    return: the number of superpixels
    '''
    interactive_SPs = lbl
    sizeInterSpList = []
    for i in np.unique(interactive_SPs)[1:]:
        separationSPs = np.where(interactive_SPs == i, i, 0)
        separationSPs = skimage.transform.resize(separationSPs, (299, 299))
        sizeInterSp = np.count_nonzero(separationSPs)
        sizeInterSpList.append(sizeInterSp)
    return (299 * 299) // np.mean(sizeInterSpList)


def get_image_with_mask(img, mask, segments, coefficients, mask_features):
    """
    Generates images and predictions in the neighborhood of this image.

    input: image: 3d numpy array, the image
           mask: the marks of numpy array [0, 1]
           segments: segmentation of the image

    return: perturbed_image
    """
    active_pixels = np.where(mask == 1)[0]
    mask = np.zeros(segments.shape)
    for idx, active in enumerate(active_pixels):
        mask[segments == active] = idx + 1
    mask_3d = np.dstack((mask,mask,mask))  # mask have to be 3D (299,299,3)

    # create positive green image
    green = np.full_like(img,(0,1,0))
    # create negative red image
    red = np.full_like(img,(1,0,0))
    result = img

    alpha = coeff_to_alpha(np.absolute(coefficients))

    mask_coeff = coefficients[mask_features]
    alpha = alpha[mask_features]
    for idx, coeff in enumerate(mask_coeff):
        if coeff >= 0:
            mask_color = green
        else:
            mask_color = red

        img_mask = cv2.addWeighted(mask_color,alpha[idx], img,0.5,gamma=0.1)   # gamma is contrast
        # combine img and img_cyan using mask
        result = np.where(mask_3d == idx + 1, img_mask, result)
    result = np.where(result > 1, 1, result)
    return result

def coeff_to_alpha(coeff):
    '''
    Alpha is depend on the rank of coefficients.
    '''
    if np.size(coeff) == 1:
        return np.array([0.7])
    step = np.arange(0.3, 1, 0.7/np.size(coeff))
    ranks = coeff.argsort()
    ranks = ranks.argsort()
    alpha = step[ranks]

    return alpha


def inter_lime(image, lbl, label_names, i_class, top_guesses):
    #i_class = 0  # set the explained class (from 0)
    image = skimage.transform.resize(image, (299, 299))
    image = (image - 0.5) * 2  # Inception pre-processing
    num_top_features = 2  # the number of top superpixels(coefficients) you want to see
    num_perturb = 150  # number of perturbed points
    perturb_art = 0  # 0 -> random; 1 -> exactly
    explained_model = keras.applications.inception_v3.InceptionV3()

    '''
    module prediction
    '''
    #np.random.seed(222)
    preds = explained_model.predict(image[np.newaxis, :, :, :], verbose = 0)
    top_pred_classes = preds[0].argsort()[-top_guesses:][::-1]

    '''
    LIME-segmentation the image
    function: slic segmentation
    '''
    # set a do-while loop to avoid the too few num_SPs
    num_SPs = get_num_segmentSPs(lbl, label_names)
    segment_SPs = skimage.segmentation.slic(image, n_segments=num_SPs, compactness=10)
    temp_num_SPs = num_SPs
    while np.unique(segment_SPs).shape[0] < num_SPs:
        temp_num_SPs = temp_num_SPs + temp_num_SPs // 2
        segment_SPs = skimage.segmentation.slic(image, n_segments=temp_num_SPs, compactness=10)


    '''
    mix segmentation from slic and interactively segmentation
    '''
    # recover superpixels with interactively segmentation
    interactive_SPs, inter_label_name = mix_segment(lbl, label_names, segment_SPs)

    final_num_SPs = np.unique(interactive_SPs).shape[0]


    '''
    perturbation
    '''

    if perturb_art == 0:
        # random perturbation
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, final_num_SPs))
        perturbations[0, :] = 1
    else:
        # exactly perturbation
        rows = 2 ** final_num_SPs
        cols = final_num_SPs
        binary_matrix = [[1 if ((i >> j) & 1) else 0 for j in range(cols)] for i in range(rows)]
        perturbations = np.array(binary_matrix)
    # new training images for LIME
    predictions = []
    for pert in perturbations:
        perturbedImage = perturb_image(image, pert, interactive_SPs)
        pred = explained_model.predict(perturbedImage[np.newaxis, :, :, :], verbose = 0)
        predictions.append(pred)
    predictions = np.array(predictions)

    # calculate the distance
    original_image = np.ones(final_num_SPs)[np.newaxis, :]  # Perturbation with all superpixels enabled
    distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
    kernel_width = 0.25
    weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

    '''
    train the explained module
    '''
    inter_sp_coeff = []
    explained_class = top_pred_classes[i_class]
    explained_class_name = decode_predictions(preds)[0][i_class][1]
    simple_model = LinearRegression()
    simple_model.fit(X=perturbations, y=predictions[:, :, explained_class], sample_weight=weights)
    coeff = simple_model.coef_[0]

    temp_str = 'Explaining prediction: ' + str(explained_class_name)
    inter_sp_coeff.append(temp_str)

    num_inter_feature = len(inter_label_name[:]) - 1
    for i in range(num_inter_feature):
        temp_str = "coefficient of label " + str(inter_label_name[i + 1]) + ": " + str(coeff[i])
        inter_sp_coeff.append(temp_str)
        #show_explain = "\n".join(inter_sp_coeff)

    top_feature = np.argsort(coeff)[-1:]
    temp_str = "coefficient of top 1 class: " + str(coeff[top_feature])
    inter_sp_coeff.append(temp_str)
    show_explain = "\n".join(inter_sp_coeff)
    return show_explain


if __name__ == "__main__":
    i_class = 0
    image_file = 'image_file/pig.JPEG'
    json_file = 'json_file/pig.json'
    explained_model = keras.applications.inception_v3.InceptionV3()
    top_guesses = 5
    lbl, label_names = test.convert_json(json_file)
    image_rgba = skimage.io.imread(image_file)
    if np.shape(image_rgba)[2] != 3:
        image = skimage.color.rgba2rgb(image_rgba)
    else:
        image = image_rgba

    image = skimage.transform.resize(image, (299, 299))
    image = (image - 0.5) * 2  # Inception pre-processing


    show_explain = inter_lime(image,lbl, label_names,0,5)
    print(show_explain)


'''
    show the results
    
    top_features = np.argsort(coeff)[-num_top_features:]
    print(top_features)
    print('coefficient of top 2 classes: ', coeff[top_features])
    num_inter_feature = len(inter_label_name[:]) - 1
    
    for i in range(num_inter_feature):
        print('coefficient of', inter_label_name[i + 1], coeff[i])

    inter_features = list(range(num_inter_feature))
    mask = np.zeros(final_num_SPs)
    mask[inter_features] = True

    plt.figure()
    skimage.io.imshow(get_image_with_mask(image / 2 + 0.5, mask, interactive_SPs, coeff, mask_features = inter_features))
    plt.show()
    '''



