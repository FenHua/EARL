import torch
import numpy as np
from Utils.vector2image import showallframes

model_name ='c3d'
i = 4   # the i-th video, there are total 5 video segments in directory TT
vid = torch.from_numpy(np.load('TT/{}.npy'.format(i)))
showallframes([],vid.data.cpu().numpy(),model_name)