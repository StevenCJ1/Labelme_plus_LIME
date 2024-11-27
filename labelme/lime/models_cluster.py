#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   models_cluster.py
@Time    :   2023/12/17
@Author  :   Chenjun Wu
@Version :   1.0
@Contact :   chenjun.wu@mailbox.tu-dresden.de
@License :   (C)Copyright 2023-2024
@Desc    :   uncertainty quantification
'''
import copy
import torch
import torchvision.models as models
from PIL import Image, ImageFilter
import numpy as np
from torchvision import transforms
import skimage.io
import skimage.color
import skimage.segmentation
from qtpy import QtCore
from qtpy.QtCore import Qt
from labelme import PY2
from labelme.lime import shape2label
from labelme.lime import explain_lime
import cv2
import sklearn.metrics
from sklearn.linear_model import LinearRegression

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import io

import matplotlib.pyplot as plt

class ModelCluster():
    def __init__(self, model_type):

        self.model_list = []
        for i in range(1,8):
            if i == 7:
                model_path = f'lime/Imagenet/resnet50-7-MumbaiButcher/{model_type}.pth.tar'
            else:
                model_path = f'lime/Imagenet/resnet50-{i}/{model_type}.pth.tar'
            model = self.get_model_state(model_path)
            self.model_list.append(model)



    def get_model_state(self, model_path):
        model = models.resnet50(pretrained=False)
        # load .pth file(It's used save() to save all model)
        total_model = torch.load(model_path, map_location=torch.device('cpu'))

        # extra state_dict
        state_dict = total_model['state_dict']

        # delete 'module.' prefix
        state_dict = {k.partition('module.')[2]: v for k, v in state_dict.items()}

        # add state_dict(saved) to the models
        model.load_state_dict(state_dict)

        # set the model as evaluation
        model.eval()
        return model

    def get_mean_preds(self, image_pred):
        output_list = []

        input_batch = image_pred.unsqueeze(0)  # creat mini-batch

        for i in range(0,7):
            # predicate
            with torch.no_grad():
                output = self.model_list[i](input_batch)
                output_list.append(output)
        outputs = torch.cat(output_list, dim=0)
        output_mean = outputs.mean(dim = 0,keepdim = True)
        # 获取概率分布
        probabilities = torch.nn.functional.softmax(output_mean[0], dim=0)

        return probabilities

    def get_mean_preds_pert(self,perturbed_image):
        output_list = []

        input_batch = perturbed_image.unsqueeze(0)   # creat mini-batch

        for i in range(7):
            # predicate
            with torch.no_grad():
                output = self.model_list[i](input_batch)
                output_list.append(output)
        outputs = torch.cat(output_list, dim=0)
        output_mean = outputs.mean(dim = 0,keepdim = True)
        # 获取概率分布
        probabilities = torch.nn.functional.softmax(output_mean[0], dim=0)

        return probabilities


    def get_random_preds(self):
        # 图像预处理
        #preprocess = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #])
        #input_image = input_file.permute(2, 0, 1).float()

        #input_tensor = preprocess(input_image)
        input_batch = self.image_pred.unsqueeze(0)  # 创建一个 mini-batch

        idx = np.random.randint(0, 6)
        with torch.no_grad():
            output = self.model_list[idx](input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return probabilities.numpy().reshape(1,-1)

    def get_oneModel_pred(self,perturbed_image, model_idx):
        prob = np.empty([0,1000])
        input_batch = perturbed_image.unsqueeze(0)  # 创建一个 mini-batch


        with torch.no_grad():
            output = self.model_list[model_idx](input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy().reshape(1,-1)
        prob = np.vstack((prob,probabilities))
        return prob

    def get_all_preds(self,perturbed_image):
        prob = np.empty([0,1000])
        input_batch = perturbed_image.unsqueeze(0)  # 创建一个 mini-batch
        for idx in range(7):
            with torch.no_grad():
                output = self.model_list[idx](input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0).numpy().reshape(1,-1)
            prob = np.vstack((prob,probabilities))
        return prob

class EnsembleThread(QtCore.QThread):
    pred_results = QtCore.Signal(str)
    def run(self):
        if bool(self.parent().pred_input_line.text()):
            self.pred_input_num = self.parent().pred_input_line.text()
            num_top_guess = int(self.pred_input_num)
        else:
            num_top_guess = 5
        new_preds_list = []
        model = ModelCluster(model_type='model_best')
        image_pred = torch.tensor([3, 224, 224], dtype=torch.float32)
        image_show = np.array([256, 256, 3], dtype=np.float32)
        # image must preprocess for Imagenet
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = Image.open(self.parent().filename)
        if input_image.mode == 'RGBA':
            input_image = input_image.convert("RGB")
        input_tensor = preprocess(input_image)
        image_pred = input_tensor
        image_np = np.asarray(input_image)
        image_show = skimage.transform.resize(image_np, (256, 256))
        mean_prediction = model.get_mean_preds(image_pred)
        with open("lime/Imagenet/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        probs, top_pred_classes = torch.topk(mean_prediction, num_top_guess)
        for i in range(probs.size(0)):
            pred = str(i + 1) + ". " + categories[top_pred_classes[i]] + "  " + str(probs[i].item())
            new_preds_list.append(pred)
            show_preds = "\n".join(new_preds_list)


        self.pred_results.emit(show_preds)

class UncLimeThread(QtCore.QThread):
    # Create a counter thread
    change_value = QtCore.Signal(int)
    finished = QtCore.Signal(str)
    seg_img = QtCore.Signal(list)
    result1_img = QtCore.Signal(list)
    result2_img = QtCore.Signal(list)
    num_pos_sps = QtCore.Signal(str)
    num_neg_sps = QtCore.Signal(str)

    def run(self):
        self.change_value.emit(1)

        input_i_class = self.parent().lime_select_input.text()

        def format_shape(s):
            data = s.other_data.copy()
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    description=s.description,
                    shape_type=s.shape_type,
                    flags=s.flags,
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.parent().labelList]

        filename = self.parent().filename
        image_rgba = Image.open(filename)

        # image_rgba = predict.convertQImageToMat(self.parent().image)

        if np.shape(image_rgba)[2] != 3:
            input_image = skimage.color.rgba2rgb(image_rgba)
        else:
            input_image = image_rgba
        image_pred = torch.tensor([3, 224, 224], dtype=torch.float32)
        image_show = np.array([256, 256, 3], dtype=np.float32)
        # image must preprocess for Imagenet
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        explained_models = ModelCluster(model_type='model_best')


        i_class = int(input_i_class) - 1  # set the explained class (from 0)
        top_guesses = 5
        input_tensor = preprocess(input_image)
        image_pred = input_tensor
        image_np = np.asarray(input_image)
        image_show = skimage.transform.resize(image_np, (256, 256))

        lbl_image = skimage.io.imread(filename)
        if np.shape(lbl_image)[2] != 3:
            lbl_image = skimage.color.rgba2rgb(lbl_image)
        else:
            lbl_image = lbl_image
        data = dict(
            shapes=shapes,
            image_numpy=lbl_image,
        )
        lbl, label_names = shape2label.convert_shapes(data)

        num_top_features = 2  # the number of top superpixels(coefficients) you want to see
        num_perturb = 150  # number of perturbed points
        perturb_art = 0  # 0 -> random; 1 -> exactly
        coeff_list = []
        # explained_model = keras.applications.inception_v3.InceptionV3()
        self.change_value.emit(2)
        '''
        module prediction
        '''
        with open("lime/Imagenet/imagenet_classes.txt", "r") as f:
            categories = [s.strip() for s in f.readlines()]
        mean_prediction = explained_models.get_mean_preds(image_pred)
        # Show top categories per image
        probs, top_pred_classes = torch.topk(mean_prediction, 5)
        prime_pred_class = top_pred_classes[0]

        '''
        LIME-segmentation the image
        function: slic segmentation
        '''
        # set a do-while loop to avoid the too few num_SPs
        num_SPs = explain_lime.get_num_segmentSPs(lbl, label_names, 256)
        segment_SPs = skimage.segmentation.slic(image_show, n_segments=num_SPs, compactness=10)
        temp_num_SPs = num_SPs
        while np.unique(segment_SPs).shape[0] < num_SPs:
            temp_num_SPs = temp_num_SPs + temp_num_SPs // 2
            segment_SPs = skimage.segmentation.slic(image_show, n_segments=temp_num_SPs, compactness=10)
        self.change_value.emit(4)
        '''
        mix segmentation from slic and interactively segmentation
        '''
        # recover superpixels with interactively segmentation
        interactive_SPs, inter_label_name = explain_lime.mix_segment(lbl, label_names, segment_SPs, 256)
        seg_img = []


        # Set the num to each SPs
        final_num_SPs = np.unique(interactive_SPs).shape[0]

        # 为每个超像素分配标号并在图像上显示

        image_with_num = copy.deepcopy(image_show)
        for (i, segVal) in enumerate(np.unique(interactive_SPs)):
            # 计算超像素区域的中心点
            mask = np.zeros(image_with_num.shape[:2], dtype="uint8")
            mask[interactive_SPs == segVal] = 255
            coords = np.column_stack(np.where(mask > 0))
            center = coords.mean(axis=0).astype("int")

            if mask[center[0], center[1]] == 0:
                random_index = np.random.randint(0, coords.shape[0])
                random_coord = coords[random_index, :]
                center = random_coord

            cv2.putText(image_with_num, str(i), (center[1], center[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 0, 0), 1)

        temp_img = (skimage.segmentation.mark_boundaries(image_with_num, interactive_SPs) * 255).astype(
            np.uint8)

        seg_img.append(temp_img)
        self.seg_img.emit(seg_img)

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

        self.change_value.emit(5)
        # new training images for LIME
        predictions = []
        for idx, pert in enumerate(perturbations):
            perturbedImage = explain_lime.unc_perturb_image(image_pred, pert, interactive_SPs)
            pred = explained_models.get_all_preds(perturbedImage)
            predictions.append(pred)

            self.change_value.emit(int((idx + 5) * 100 / (num_perturb + 10)))
        predictions = np.array(predictions)

        for i in range(100):
            # get the random prediction
            random_pred_idx = np.random.randint(0, 7, size=(150))
            random_pred = np.empty((num_perturb, 1, 1000))
            for i in range(num_perturb):
                random_pred[i, 0, :] = predictions[i, random_pred_idx[i], :]

            # calculate the distance
            original_image = np.ones(final_num_SPs)[np.newaxis, :]  # Perturbation with all superpixels enabled
            distances = sklearn.metrics.pairwise_distances(perturbations, original_image, metric='cosine').ravel()
            kernel_width = 0.25
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))

            '''
            train the explained module
            '''
            explained_class = top_pred_classes[i_class]
            explained_class_name = categories[explained_class]

            simple_model = LinearRegression()
            simple_model.fit(X=perturbations, y=random_pred[:, :, explained_class], sample_weight=weights)
            coeff = simple_model.coef_[0]

            coeff_list.append(coeff)
        coeff_np = np.array(coeff_list)

        '''
        train the explained module
        '''
        inter_sp_coeff = []
        coeff = np.mean(coeff_np,0)

        self.num_pos_sps.emit(str(np.sum(coeff >= 0)) + ')')
        self.num_neg_sps.emit(str(np.sum(coeff < 0)) + ')')

        temp_str = 'Explaining prediction: ' + str(explained_class_name)
        inter_sp_coeff.append(temp_str)

        num_inter_feature = len(inter_label_name[:]) - 1
        for i in range(num_inter_feature):
            temp_str = "coefficient of label " + str(inter_label_name[i + 1]) + ": " + str(coeff[i])
            inter_sp_coeff.append(temp_str)
            # show_explain = "\n".join(inter_sp_coeff)

        top_feature = np.argsort(coeff)[-2:]
        all_feature = np.argsort(coeff)[:]

        temp_str = "coefficient of all classes: " + str(coeff[all_feature])
        inter_sp_coeff.append(temp_str)
        explain_result = "\n".join(inter_sp_coeff)

        self.finished.emit(explain_result)
        self.change_value.emit(100)




        def coeff_to_alpha(coeff):
            '''
            Alpha is depend on the rank of coefficients.
            '''
            if np.size(coeff) == 1:
                return np.array([0.7])
            step = np.arange(0.3, 1, 0.7 / np.size(coeff))
            ranks = coeff.argsort()
            ranks = ranks.argsort()
            alpha = step[ranks]

            return alpha

        alpha = coeff_to_alpha(np.absolute(coeff))




        result1_img = []
        # 转换成 DataFrame
        data_df = pd.DataFrame(coeff_np)

        data_long = data_df.melt(var_name='Coeff', value_name='Value')
        fig = Figure()
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        sns.violinplot(x="Coeff", y="Value", data=data_long, ax = ax)
        canvas.draw()  # 绘制图表

        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)

        image = Image.open(buf)
        image_np = np.array(image)
        image_np = image_np[:, :, :3]
        buf.close()

        result1_img.append(image_np)
        self.result1_img.emit(result1_img)




        mask = np.zeros(final_num_SPs)
        mask[all_feature] = True
        result2_img = []
        # int_img = ((image / 2 + 0.5) * 255).astype(np.uint8)
        mask_img = explain_lime.get_image_with_mask(image_show, mask, interactive_SPs, coeff, boundary=True,
                                                    alpha=alpha)
        result2_img.append(mask_img)
        # result2_img.append((mask_img * 255).astype(np.uint8))
        self.result2_img.emit(result2_img)


