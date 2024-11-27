#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   uncertainty_quantification.py
@Time    :   2023/12/17
@Author  :   Chenjun Wu
@Version :   1.0
@Contact :   chenjun.wu@mailbox.tu-dresden.de
@License :   (C)Copyright 2023-2024
@Desc    :   uncertainty quantification
'''
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms


class ModelCluster():
    def __init__(self, num_model):
        self.image_pred = torch.tensor([3,224,224], dtype = torch.float32)
        self.image_show = np.array([256,256,3], dtype = np.float32)
        self.model_list = []

        if num_model == 7:
            model_path = 'Imagenet/resnet50-7-MumbaiButcher/model_best.pth.tar'
        else:
            model_path = f'Imagenet/resnet50-{num_model}/model_best.pth.tar'
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

    def get_mean_preds(self, input_file):
        output_list = []
        # 图像预处理
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_image = input_file
        #input_image = Image.open('dataset/image_file/basktballplayer.JPEG')
        input_tensor = preprocess(input_image)
        self.image_pred = input_tensor
        input_batch = input_tensor.unsqueeze(0)  # 创建一个 mini-batch

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

    def get_random_preds(self, input_tensor):
        # 图像预处理
        #preprocess = transforms.Compose([
        #    transforms.Resize(256),
        #    transforms.CenterCrop(224),
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #])
        #input_image = input_file.permute(2, 0, 1).float()

        #input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # 创建一个 mini-batch

        idx = np.random.randint(0, 6)
        with torch.no_grad():
            output = self.model_list[idx](input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        return probabilities.numpy().reshape(1,-1)

'''
import torch
from torchvision import transforms

# 假设你有一个PyTorch张量表示图像，形状为[channels, height, width]
torch_tensor = torch.rand(3, 256, 256)

# 创建一个transform来调整图像
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 应用transform
torch_tensor = preprocess(torch_tensor)

# 确保张量的形状是[batch_size, channels, height, width]
# 如果你的张量只有一个图像，你可能需要使用unsqueeze来添加一个批次维度
torch_tensor = torch_tensor.unsqueeze(0)

'''