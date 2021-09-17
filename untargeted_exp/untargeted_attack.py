import sys
import torch
import numpy as np
from Utils.utils import *
import torch.nn as nn
from attack.untargetedAttack import attack
# C3D_K_Model return K results (default K=1)
from model_wrapper.vid_model_top_k import C3D_K_Model,\
    LRCN_K_Model,I3D_K_Model

gpus = [0]      # GPU setting
len_limit = 4
image_models = ['resnet50']
multiple_gpus = len(gpus) > 1  # multi gpus
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])  # The GPUs are visible to the system
model_name = 'lrcn'
dataset_name = 'hmdb51'

# ---------------------------start-----------------------------------------------
print('load {} dataset'.format(dataset_name))
test_data = generate_dataset(model_name, dataset_name)  # get testing dataset
print('load {} model'.format(model_name))
model = generate_model(model_name, dataset_name)        # get a recognition model
try:
    model.cuda()   # GPU
except:
    pass
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
elif model_name == 'i3d':
    vid_model = I3D_K_Model(model)
else:
    vid_model = LRCN_K_Model(model)
# gets ids of the samples to be attacked
attacked_ids = get_attacked_samples(model, test_data, 100, model_name, dataset_name)
x0, label = test_data[attacked_ids[1]]  # test id=7
x0 = image_to_vector(model_name, x0)
vid = x0.cuda()
res, iter_num, adv_vid = attack(vid_model, vid, label[1], len_limit, image_models, gpus)
if res:
    # success
    AP = pertubation(vid, adv_vid)
    print('untargeted attack succeed using {} quries'.format(iter_num))
    print('The average pertubation of video is: {}'.format(AP.cpu()))
else:
    # fail
    print('--------------------Attack Fails-----------------------')