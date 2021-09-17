import torch
import random
import numpy as np
from torch.optim import lr_scheduler
from attack.group_generator import EquallySplitGrouping
from adapselection.Agent import feature_extractor,frames_select
from adapselection.Reward import untargeted_reward,targeted_reward
from adapselection.utils import finelist,agent_output,sparse_perturbation


# Gradient estimation of targeted attacks (NES)
def TargetedNES(model, vid,target_class, n, sigma, sub_num):
    # n is the sample number for NES, sub_num is also the sample number and used to
    # prevent GPU resource insufficiency if n is too large
    with torch.no_grad():
        grads = torch.zeros(vid.size()).cuda()               # gradient initialization
        count_in = 0                                         # record the iteration number
        loss_total = 0                                       # loss per NES
        batch_loss = []                                      # loss per batch
        batch_noise = []                                     # noise per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))       # initialize sub_num samples
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma   # half noise
            all_noise = torch.cat([noise_list, -noise_list], 0)                            # total noise
            adv_vid_rs += all_noise                                              # input samples
            top_val, top_idx, logits = model(adv_vid_rs)                                   # the classification results
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(
                target_class, dtype=torch.long,device='cuda').repeat(sub_num),reduction='none')  # loss
            batch_loss.append(loss)
            batch_noise.append(all_noise)
        # batch processing
        batch_noise = torch.cat(batch_noise, 0)
        batch_loss = torch.cat(batch_loss)
        valid_loss = batch_loss                        # total loss
        loss_total += torch.mean(valid_loss).item()    # total loss
        count_in += valid_loss.size(0)                 # valid number
        noise_select = batch_noise                     # noise
        grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)
        if count_in == 0:
            return None, None
        # return estimated gradient and loss
        return loss_total / count_in, grads


# untargeted attack gradient estimation (NES)
def sim_rectification_vector(model, vid, tentative_directions, n, sigma, target_class,
                             rank_transform, sub_num, group_gen, untargeted):
    # n is the sample number for NES, sub_num is also the sample number and used to
    # prevent GPU resource insufficiency if n is too large
    with torch.no_grad():
        grads = torch.zeros(len(group_gen), device='cuda')   # Partitioned gradient initialization
        count_in = 0                                         # record the effective number
        loss_total = 0                                       # loss per NES
        batch_loss = []                                      # loss per batch
        batch_noise = []                                     # noise per batch
        batch_idx = []                                       # category per batch
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size()))             # sample initialization
            noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma   # noise initialization
            all_noise = torch.cat([noise_list, -noise_list], 0)                            # noise
            perturbation_sample = group_gen.apply_group_change(tentative_directions, all_noise)  # produce perturbations
            adv_vid_rs += perturbation_sample                                               # add perturbations to input
            del perturbation_sample                                                         # release gpu resources
            top_val, top_idx, logits = model(adv_vid_rs)                                    # classification results
            if untargeted:
                loss = -torch.max(logits, 1)[0]       # loss function (reduce the confidence of the maximum probability)
            else:
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(target_class, dtype=torch.long,
                                                                              device='cuda').repeat(sub_num),
                                                         reduction='none')
            batch_loss.append(loss)
            batch_idx.append(top_idx)
            batch_noise.append(all_noise)
        # concat operations
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)
        # sorting loss
        if rank_transform:
            good_idx = torch.sum(batch_idx == target_class, 1).byte()                               # valid sample
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))    # penalty
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')                           #
            sort_index = changed_loss.sort()[1]                                        # sort loss to obtain coordinates
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
            available_number = torch.sum(good_idx).item()                              # the number of valid samples
            count_in += available_number                                               # accumulative count
            unavailable_number = n - available_number                                               # invalid number
            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                       loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')
            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)           # weighted gradient
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),0)
        else:
            idxs = (batch_idx == target_class).nonzero()   # valid samples
            valid_idxs = idxs[:, 0]                        # coordinates
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)     # valid loss
            loss_total += torch.mean(valid_loss).item()    # average loss
            count_in += valid_loss.size(0)                 # count
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)  # valid noise
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),0)
        if count_in == 0:
            return None, None
        # return estimated gradient and loss
        return loss_total / count_in, grads


# untargeted video attacks. The input video should be a tensor [num_frames, C, W, H]
# Input should be normalized to [0,1]
def untargeted_video_attack(vid_model, vid, directions_generator, ori_class, rank_transform=False,
                            eps=0.05,max_lr=1e-2,min_lr=1e-3, sample_per_draw=48,max_iter=30000,
                            sigma=1e-6, sub_num_sample=12,image_split=8,len_limit=2):
    num_iter = 0                                                                       # the number of iterations
    '''------------------------------agent initialization------------------------------'''
    print('Initializing the agent......')
    agent = frames_select()                                                            # agent(Bidirectional lstm)
    GetFeatures = feature_extractor()                                                  # feature extractor
    optimizer = torch.optim.Adam(agent.parameters(),lr=1e-05,weight_decay=1e-05)       # optimizer
    scheduler = lr_scheduler.StepLR(optimizer,step_size=30,gamma=0.1)                  # learning rate setting
    agent = agent.cuda()
    baseline=0.0
    '''---------------add random perturbations to len_limit video frames---------------'''
    perturbation = (torch.rand_like(vid) * 2 - 1) * eps                                # initial perturbations
    key_list = random.sample(range(0,16),len_limit)               # generate a len_limit length frame sequence
    perturbation = sparse_perturbation(perturbation,key_list)
    adv_vid = torch.clamp(vid.clone() + perturbation, 0., 1.)         # initialize an sparse adversarial video
    cur_lr = max_lr                                                                    # current learning rate
    last_p = []                                                                        # probability value
    last_score = []                                                                    # probability value
    group_gen = EquallySplitGrouping(image_split)                                      # division
    keylen=[]                                  # record the coordinates of the frame that the perturbations are added to
    while num_iter < max_iter:

        '''---------------key frame selection and the corresponding mask generation----------------'''
        features = GetFeatures(adv_vid)         # video frame feature（512 dimensions）
        probs,Sidx,pre_action = agent_output(agent,features)
        beta = 1.0
        alpha = len_limit/16.0
        cost = beta * (probs.mean() - alpha) ** 2
        '''--------------------------attack--------------------------------'''
        top_val, top_idx, _ = vid_model(adv_vid[None,:])                               # model prediction results
        num_iter += 1                                                                  # query number update
        if ori_class != top_idx[0][0]:
            # if the category is inconsistent with the original category, attack succeed
            print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid,len(keylen)
        idx = (top_idx == ori_class).nonzero()                                 # coordinates of the category
        pre_score = top_val[0][idx[0][1]]                                      # the corresponding probability value
        del top_val
        del top_idx
        print('cur target prediction: {}'.format(pre_score))
        last_score.append(float(pre_score))                         # The queue records the predicted probability values
        last_score = last_score[-200:]                              # take the first 200
        if last_score[-1] >= last_score[0] and len(last_score) == 200:
            # after 200 iterations, the last predicted probability value is greater than the first probability value
            print('FAIL: No Descent, Stop iteration')               # attack fail
            return False, pre_score.cpu().item(), adv_vid,len(keylen)
        last_p.append(float(pre_score))                             # record probability values
        last_p = last_p[-20:]                                       # take the latest 20 probability values
        if last_p[-1] <= last_p[0] and len(last_p) == 20:
            # annealing operation
            if cur_lr > min_lr:
                # print("[log] Annealing max_lr")
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []
        tentative_directions = directions_generator(adv_vid).cuda()   # perturbation direction
        group_gen.initialize(tentative_directions)                    # direction division
        # gradient estimation
        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        ori_class, rank_transform, sub_num_sample, group_gen, untargeted=True)
        if l is None and g is None:
            print('nes sim fails, try again....')
            continue
        # modify gradient direction
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))
        '''----------------------compute reward and get keyframes-----------------------'''
        final_key = []
        best_reward = 0.0
        epis_rewards = []
        for _ in range(5):
            key_list = pre_action.sample()                # sample
            key = finelist(Sidx,key_list,len_limit)       # Retain the maximum probability len_limit frames
            reward = untargeted_reward(vid_model,adv_vid,key,rectified_directions,cur_lr)  # compute reward value
            reward = reward.cpu().data
            reward = reward[0]
            log_probs = pre_action.log_prob(key_list)
            expected_reward = log_probs.mean() * (reward - baseline)
            cost -= expected_reward
            epis_rewards.append(reward)
            if reward > best_reward:
                final_key = key
                best_reward = reward
        #print('The reward of this epoch is {}'.format(np.mean(epis_rewards)))
        '''-----------------------update agent-------------------------------------'''
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(),5.0)
        optimizer.step()
        baseline = 0.9*baseline + 0.1*np.mean(epis_rewards)

        del tentative_directions                # release resources
        # cumulate query number，10 is used to calculate the reward value and can be optimized to 6
        num_iter += sample_per_draw + 10
        proposed_adv_vid = adv_vid
        assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
        # PGD
        keylen = keylen+final_key                                                     # add key frames
        keylen = list(set(keylen))
        # only add perturbations to keyframes
        rectified_directions = sparse_perturbation(rectified_directions,final_key)
        proposed_adv_vid += cur_lr * rectified_directions                             # update adversarial video
        # clip
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()
        #print('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))               # Comment in batch
    return False, pre_score.cpu().item(), adv_vid,len(keylen)

# untargeted video attacks. The input video should be a tensor [num_frames, C, W, H]
# Input should be normalized to [0,1]
def targeted_video_attack(vid_model, vid, target_class, eps=0.05,max_lr=1e-2,min_lr=1e-3,
                          sample_per_draw=48,max_iter=60000, sigma=1e-6, sub_num_sample=12,
                          len_limit=2):
    '''------------------------------agent initialization------------------------------'''
    print('Initializing the agent......')
    agent = frames_select()            # agent(Bidirectional LSTM)
    GetFeatures = feature_extractor()  # feature extractor
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-05, weight_decay=1e-05)  # optimizer
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)             # learning rate setting
    agent = agent.cuda()
    baseline = 0.0
    num_iter = 0                                           # the number of iterations
    adv_vid = vid.clone()                                  # initialize an adversarial video
    cur_lr = max_lr                                        # current learning rate
    last_loss = []                                         # record loss value
    g = 0.0                                                # initialize g value
    keylen = []          # record the coordinates of the frame that the perturbations are added to
    while num_iter < max_iter:
        '''---------------key frame selection and the corresponding mask generation----------------'''
        features = GetFeatures(adv_vid)  # video frame feature（512 dimensions）
        probs, Sidx, pre_action = agent_output(agent, features)
        beta = 1.0
        alpha = len_limit / 16.0
        cost = beta * (probs.mean() - alpha) ** 2
        '''--------------------------attack--------------------------------'''
        top_val, top_idx, _ = vid_model(adv_vid[None,:])   # model prediction results
        num_iter += 1                                      # query number update
        if target_class == top_idx[0][0]:
            # if the category is consistent with the targeted category, attack succeed
            print('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid, len(keylen)
        pre_g = g                                          # Record the last gradient
        # nes gradient estimation
        l, g = TargetedNES(vid_model,adv_vid,target_class,sample_per_draw,sigma,sub_num_sample)
        if l is None and g is None:
            print('nes sim fails, try again....')
            continue
        g = 0.5*pre_g + 0.5*g                              # momentum，momentum magnitude is 0.5
        last_loss.append(l)                                # record loss
        last_loss = last_loss[-5:]                         # Take the latest five loss value
        # adjust the learning rate
        if last_loss[-1] > last_loss[0]:
            if cur_lr > min_lr:
                # print('[log] Annealing max_lr')          # to save time
                cur_lr = max(cur_lr/2.,min_lr)
            last_loss=[]                                   # clear loss statistics

        # ---------------------------Agent learning and update------------------------------
        final_key = []
        best_reward = 0.0
        epis_rewards = []
        for _ in range(5):
            key_list = pre_action.sample()             # sample
            key = finelist(Sidx, key_list, len_limit)  # Retain the maximum probability len_limit frames
            reward = targeted_reward(vid_model, adv_vid, key, g, cur_lr,target_class)   # compute reward value
            reward = reward.cpu().data
            reward = reward[0]
            log_probs = pre_action.log_prob(key_list)
            expected_reward = log_probs.mean() * (reward - baseline)
            cost -= expected_reward
            epis_rewards.append(reward)
            if reward > best_reward:
                final_key = key
                best_reward = reward
        # Cumulative number of queries,10 is used to calculate the reward value and can be optimized to 6
        num_iter += sample_per_draw + 10
        # print('The reward of this epoch is {}'.format(np.mean(epis_rewards)))
        '''-----------------------update agent-------------------------------'''
        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 5.0)
        optimizer.step()
        baseline = 0.9 * baseline + 0.1 * np.mean(epis_rewards)
        keylen = keylen+final_key                          # add key frames
        keylen = list(set(keylen))
        g = sparse_perturbation(g,final_key)
        proposed_adv_vid = adv_vid
        # PGD
        proposed_adv_vid -= cur_lr * torch.sign(g)         # update adversarial video
        # clip
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()
        print('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))  # save time
    return False, num_iter, adv_vid, len(keylen)