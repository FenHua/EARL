from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datasets.hmdb51 import HMDB51

# Path setting
V_P_HMDB = 'datasets/data/HMDB51/hmdb51-jpg'
V_A_HMDB = 'datasets/data/HMDB51/hmdb51-annotation/hmdb51_1.json'


# Gets the validation dataset
def get_validation_set(dataset,spatial_transform, temporal_transform, target_transform):
    if dataset == 'hmdb51':
        video_path=V_P_HMDB
        annotation_path=V_A_HMDB
        validation_data = HMDB51(
            video_path,
            annotation_path,
            'validation',
            1,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=16)
    return validation_data