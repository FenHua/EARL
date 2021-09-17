from attack.attack_utils import targeted_video_attack


# targeted_attack
def attack(vid_model,vid,target_label,len_limit):
    res, iter_num, adv_vid, keylen= targeted_video_attack(vid_model, vid, target_label,len_limit=len_limit)
    # attack state, query number, adv_vid, key_frame number
    return res,iter_num,adv_vid,keylen