import os
import sys
import torch
import torch.nn as nn
from torchvision import models
# feature extractor for generating initial perturbations
from model_wrapper.image_model_wrapper import ResNetFeatureExtractor, \
    DensenetFeatureExtractor,TentativePerturbationGenerator
# attack method
from attack.attack_utils import untargeted_video_attack
# some code is referred to https://github.com/Jack-lx-jiang/VBAD


def attack(vid_model,vid,vid_label,len_limit,image_models,gpus):
    layer = ['fc']             # output of full layer
    extractors = []            # logits
    multiple_gpus = len(gpus) > 1
    random_mask=0.9
    if multiple_gpus:
        # multi-gpu
        advs_device = list(range(1, len(gpus)))
    else:
        advs_device = [0]
    # The alternative model is used to initialize the perturbation
    if 'resnet50' in image_models:
        resnet50 = models.resnet50(pretrained=True)
        resnet50_extractor = ResNetFeatureExtractor(resnet50, layer).eval().cuda()
        if multiple_gpus:
            resnet50_extractor = nn.DataParallel(resnet50_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(resnet50_extractor)
    if 'densenet121' in image_models:
        densenet121 = models.densenet121(pretrained=True).eval()
        densenet121_extractor = DensenetFeatureExtractor(densenet121, layer).eval().cuda()
        if multiple_gpus:
            densenet121_extractor = nn.DataParallel(densenet121_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(densenet121_extractor)
    if 'densenet169' in image_models:
        densenet169 = models.densenet169(pretrained=True).eval()
        densenet169_extractor = DensenetFeatureExtractor(densenet169, layer).eval().cuda()
        if multiple_gpus:
            densenet169_extractor = nn.DataParallel(densenet169_extractor, advs_device).eval().cuda(advs_device[0])
        extractors.append(densenet169_extractor)
    # Initialize the perturbation generator
    directions_generator = TentativePerturbationGenerator(extractors, part_size=32, preprocess=True,
                                                          device=advs_device[0])

    # un-targeted attack
    directions_generator.set_untargeted_params(vid.cuda(), random_mask, scale=5.)  # parameter setting
    res, iter_num, adv_vid,keynum = untargeted_video_attack(vid_model, vid, directions_generator, vid_label,
                                                     rank_transform=True,len_limit=len_limit)
    # attack state, query number, adv_vid, key_frame number
    return res,iter_num,adv_vid,keynum