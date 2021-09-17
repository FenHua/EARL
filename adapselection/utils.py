import torch
import random
import numpy as np
from torch.distributions import Bernoulli


# select "limit_len" lengths of key frames with high probability
def finelist(Sidx,key_list,limit_len):
    '''
        Sidx: Sorted index
        key_list: key frames agent selected
        limit_len: The upper limit of key frames
    '''
    _actions = key_list.detach()                       # prevent parameter updates
    masklist = _actions.squeeze().nonzero().squeeze()  # the coordinate of the non-zero element
    key = []
    F = 0
    for i in Sidx:
        if F>limit_len:
            break
        if i in masklist:
            key.append(i)
            F = F+1
    return key


# key frame selection by agent
def agent_output(agent,features):
    probs = agent(features[None,:])        # probability value per frame
    t_probs = probs.data.cpu().squeeze().numpy()
    pidx = np.argsort(t_probs).tolist()
    actions = Bernoulli(probs)             # bernoulli function
    return probs, pidx, actions


# sparse perturbations
def sparse_perturbation(perturbation,key):
    MASK = torch.zeros(perturbation.size())  # initialization of the mask
    MASK[key, :, :, :] = 1
    sparse_perturbation = perturbation * (MASK.cuda())
    return sparse_perturbation