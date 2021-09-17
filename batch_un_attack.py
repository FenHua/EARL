import sys
import torch
import argparse
import numpy as np
from Utils.utils import *
from attack.untargetedAttack import attack
# C3D_K_Model return top K results (default K=1)
from model_wrapper.vid_model_top_k import C3D_K_Model


config =argparse.ArgumentParser()
config.add_argument('--model_name',type=str,default='c3d',
                    help='The action recognition')
config.add_argument('--dataset_name',type=str,default='hmdb51',
                    help='The dataset: hmdb51/ucf101')
config.add_argument('--gpus',nargs='+',type=int,required=True,
                    help='The gpus to use')
config.add_argument('--len_limit',type=int,default=1,
                    help='The length limitation')
config.add_argument('--test_num',type=int,default=50,
                    help='The number of testing')
args = config.parse_args()

len_limit = args.len_limit
gpus = args.gpus           # GPU setting
image_models = ['resnet50']
os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join([str(gpu) for gpu in gpus])  # The GPUs are visible to the system
model_name = args.model_name
dataset_name = args.dataset_name
test_num = args.test_num

# ---------------------------start-----------------------------------------------
print('load {} model and {} dataset'.format(model_name,dataset_name))
test_data = generate_dataset(model_name, dataset_name)  # get testing dataset
model = generate_model(model_name, dataset_name)        # get a recognition model
model.cuda()  # GPU
if model_name == 'c3d':
    vid_model = C3D_K_Model(model)
# gets ids of the samples to be attacked
attacked_ids = get_samples(model_name, dataset_name)
def GetPairs(test_data,idx):
    x0, label = test_data[attacked_ids[idx]]
    x0 = image_to_vector(model_name, x0)
    return x0.cuda(),label[1]

result_root = 'untargeted_exp/results/{}_{}'.format(model_name,dataset_name)
av_metric = os.path.join(result_root, 'Avmetric.txt')
success_num = 0
total_P_num = 0.0
total_key_num = 0
total_query_num = 0.0
# HMDB-51, 50 videos are used to test
for i in range(test_num):
    output_path = os.path.join(result_root, 'vid-{}'.format(attacked_ids[i]))
    os.mkdir(output_path)
    vid,vid_label = GetPairs(test_data,i)
    res, iter_num, adv_vid,keylen = attack(vid_model, vid, vid_label, len_limit, image_models, gpus)
    if res:
        AP = pertubation(vid, adv_vid)
        total_query_num += iter_num
        total_P_num += AP
        success_num += 1
        total_key_num += keylen
        metric_path = os.path.join(output_path, 'metric.txt')  # record the results
        adv_path = os.path.join(output_path, 'adv_vid.npy')
        np.save(adv_path, adv_vid.cpu().numpy())
        f = open(metric_path, 'w')
        f.write(str(iter_num))
        f.write('\n')
        f.write(str(str(AP.cpu())))
        f.write('\n')
        f.close()
        f1 = open(av_metric, 'a')
        f1.write(str(total_query_num))
        f1.write('\n')
        f1.write(str(total_P_num))
        f1.write('\n')
        f1.write(str(keylen))
        f1.write('\n')
        f1.close()
    else:
        print('--------------------Attack Fails-----------------------')
f1 =open(av_metric,'a')
f1.write('--------------')
f1.write(str(success_num))
f1.write('\n')
f1.write(str(total_key_num))
f1.close()