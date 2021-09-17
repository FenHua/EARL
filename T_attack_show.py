import sys
import torch
import numpy as np
from Utils.utils import *
from attack.targetedAttack import attack
# C3D_K_Model return K results (default K=1)
from model_wrapper.vid_model_top_k import C3D_K_Model

# attack parameter setting
len_limit = 8         # Frames per attack,the bigger the fast
gpus = [0]            # GPU environment setting
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])  # The GPUs are visible to the system
model_name = 'c3d'         # model name
dataset_name = 'hmdb51'    # dataset name
# ---------------------------start-----------------------------------------------
print('load {} model and {} dataset'.format(model_name,dataset_name))
model = generate_model(model_name, dataset_name)  # get a recognition model
model.cuda()  # GPU
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
# action name
ids_labels = ["brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs",
 "dive", "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac",
 "golf", "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
 "pour", "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands",
 "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault", "stand",
 "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave"]

# attack
i = 3   # attack the i-th video, there are total 5 video segments in directory TT
vid = torch.from_numpy(np.load('TT/{}.npy'.format(i)))
vid = vid.cuda()
origin_label = vid_model(vid[None,:])[1][0,0]   # the clean video label
target_label = origin_label+1
print('The targeted label is: {}'.format(ids_labels[target_label]))
res, iter_num, adv_vid,keylen = attack(vid_model, vid, target_label, len_limit)
if res:
    print('Attack Successes!')
    label_P = vid_model(adv_vid[None,:])[1][0,0]
    print('The adversarial label is: {}'.format(ids_labels[label_P]))
else:
    print('--------------------Attack Fails-----------------------')
from Utils.vector2image import showallframes
showallframes(vid.data.cpu().numpy(),adv_vid.data.cpu().numpy(),model_name)