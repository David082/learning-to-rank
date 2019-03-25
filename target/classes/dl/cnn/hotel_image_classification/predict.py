# -*- coding: utf-8 -*-
"""
describe : 
author : yu_wei
created on : 2018/12/11
version :
refer :
-- 加载类
https://discuss.pytorch.org/t/error-loading-saved-model/8371
-- 增加一维
https://blog.csdn.net/gdymind/article/details/82933534
-- cuda
Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

class_dict = {0: u'外观', 1: u'公共区域', 2: u'健身房', 3: u'餐饮', 4: u'房间（含床）',
              5: u'房间（不含床）', 6: u'卫生间', 7: u'厨房设施', 8: u'宴会会议',
              9: u'室内泳池', 10: u'室外泳池', 11: u'周边眺望', 12: u'其他'}


def predict(model_path, image_path):
    model = torch.load(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(), normalize])
    data = Image.open(image_path)
    data = test_transforms(data)
    data = data.unsqueeze(0)  # add n
    if torch.cuda.is_available():
        data = data.cuda()
    output = model(data)
    pred = output.argmax().item()

    return output, pred


if __name__ == '__main__':
    import numpy as np

    output = np.array([[-1.4103, 2.1681, -0.4765, 1.4213, -0.1388, 0.4539, -0.3041,
                        0.1462, -0.2416, -1.1863, -1.3742, -1.1609, 1.6641]])
    output, pred = torch.from_numpy(output)
    label = class_dict.get(pred)
