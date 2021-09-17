import torch


# reward function
def untargeted_reward(model,adv_vid,key,rectified_directions,cur_lr):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    ref_adv_vid = adv_vid+cur_lr * rectified_directions     # dense attack
    MASK = torch.zeros(adv_vid.size())                      # Initialization of the mask
    MASK[key, :, :, :] = 1                                  # The key mask is assigned to 1
    proposed_adv_vid = adv_vid+cur_lr * rectified_directions * (MASK.cuda())   # Sparse attack
    # Get the logits and calculate the corresponding loss function
    _,_,logits0 = model(ref_adv_vid[None,:])
    _,_,logits1 = model(proposed_adv_vid[None,:])
    loss0 = -torch.max(logits0, 1)[0]
    loss1 = -torch.max(logits1,1)[0]
    reward = torch.exp(-torch.abs(loss0-loss1))
    return reward


# reward function
def targeted_reward(model,adv_vid,key,
                    rectified_directions,cur_lr,target_class):
    '''
    Reward values are drived from：
            Sparse attack is as close as possible to the effect of dense attack
    '''
    ref_adv_vid = adv_vid+cur_lr * rectified_directions   # dense attack
    MASK = torch.zeros(adv_vid.size())    # Initialization of the mask
    MASK[key, :, :, :] = 1                # The key mask is assigned 1
    proposed_adv_vid = adv_vid+cur_lr * rectified_directions * (MASK.cuda())   # Sparse attack
    # Get the logits and calculate the corresponding loss function
    _,_,logits0 = model(ref_adv_vid[None,:])
    _,_,logits1 = model(proposed_adv_vid[None,:])
    # Cross Entropy loss is used to evaluate the similarity of two results
    loss0 = torch.nn.functional.cross_entropy(logits0, torch.tensor(
        target_class, dtype=torch.long,device='cuda').repeat(1),reduction='none')
    loss1 = torch.nn.functional.cross_entropy(logits1, torch.tensor(
        target_class, dtype=torch.long,device='cuda').repeat(1),reduction='none')
    reward = torch.exp(-torch.abs(loss0-loss1))
    return reward