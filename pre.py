import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
from model import SqueezeNet
import torch

# 确定设备
device = torch.device("cpu")

data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open("D:\code\SqueezeNet\data\PALM-Validation400\V0079.jpg")
plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, dim=0).to(device)  # 将图像移动到设备上

name = ['非病理性近视', '病理性近视', '类别3', '类别4']  # 确保类别名称与训练时一致
model_weight_path = r"D:\code\SqueezeNet\best_model.pth"
model = SqueezeNet(num_classes=4).to(device)  # 确保类别数与训练时一致
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()

with torch.no_grad():
    output = torch.squeeze(model(img))

    predict = torch.softmax(output, dim=0)
    # 获得最大可能性索引
    predict_cla = torch.argmax(predict).numpy()
    print('索引为', predict_cla)

print('预测结果为：{},置信度为: {}'.format(name[predict_cla], predict[predict_cla].item()))
plt.show()