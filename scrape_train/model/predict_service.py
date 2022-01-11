import sys
import os
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
import torch.nn as nn
import numpy as np
import torch
from PIL import Image
from train.resnet import resnet34, resnet101, load
from torchvision import transforms


class service:
    def __init__(self, model_name, weight_path, label2name_path) -> None:
        if model_name == "resnet34":
            self.model = resnet34()
        elif model_name == "resnet101":
            self.model = resnet101()
        inchannel = self.model.fc.in_features
        self.label2name = load(label2name_path)
        self.label2name = dict(zip(
            list(self.label2name.values()),
            list(self.label2name.keys())
        ))
        self.model.fc = nn.Linear(inchannel, len(self.label2name))
        missing_keys, unexpected_keys = self.model.load_state_dict(torch.load(weight_path), strict=False)
        self.get_transform()
        print("model initialization done!")

    def get_transform(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # define transforms
        valid_transform = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              normalize])
        self.transform = valid_transform

    def predict(self, img_path, topk=3):
        with torch.no_grad():
            self.model.eval()
            img = Image.open(img_path)
            img = self.transform(img=img)
            img = torch.unsqueeze(img, dim=0)  # 添加一维
            output = torch.squeeze(self.model(img))
            predict = torch.softmax(output, dim=0).numpy()
            top_idx = np.argsort(predict)[::-1][:topk]
            res = [{self.label2name[top_i]: predict[top_i]} for top_i in top_idx]
            print(res)
        return res
        debug_stop = 1


if __name__ == "__main__":
    model_service = service(model_name="resnet34",
                            weight_path="../model_weight/restnet_resnet34.pt",
                            label2name_path="../model_weight/label2name_resnet34.pkl"
                            )

    res = model_service.predict("../test_images/Coralfish.40.jpg")

