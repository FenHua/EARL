import torch
import torch.nn as nn
from torchvision import models


# To extract features per frame (512 dimensions)
class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        # ResNet18 extract features from each video frame
        resnet = models.resnet18(pretrained=True)
        resnet.float()
        resnet.cuda()
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[: -2])
        self.pool5 = module_list[-2]

    def forward(self, vid):
        x = vid.clone()
        # IMAGENET data preprocessing
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                            device=x.get_device())[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                           device=x.get_device())[None, :, None, None]
        x = x.sub_(mean).div_(std)     # normalize
        res_conv5 = self.conv5(x)
        res_pool5 = self.pool5(res_conv5)
        res_pool5 = res_pool5.view(res_pool5.size(0), -1)
        return res_pool5


class frames_select(nn.Module):
    # Use Bidirectional Recurrent Neural Network for video frame selection
    def __init__(self):
        super(frames_select, self).__init__()
        self.rnn = nn.LSTM(512, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128*2, 1)   # hidden layer size is 128, bidirectional size is 256

    def forward(self, x):
        h, _ = self.rnn(x)
        p = torch.sigmoid(self.fc(h))
        return p