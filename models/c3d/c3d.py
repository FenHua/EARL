#from .hmdb51_opts import parse_opts as hmdb51_c3d_opts
#from .ucf101_opts import parse_opts as ucf101_c3d_opts

from .generate_models import generate_model as c3d_gen_model
import pickle


class DictToAttr(object):
    def __init__(self, args):
        for i in args.keys():
            setattr(self, i, args[i])


def generate_model_c3d(dataset):
    assert dataset in ['hmdb51','ucf101']
    if dataset == 'hmdb51':
        with open('models/c3d/hmdb51_params.pkl', 'rb') as ipt:
            model_opt = pickle.load(ipt)
        model_opt = DictToAttr(model_opt)   # to get parameter setting dictionary
    model, parameters = c3d_gen_model(model_opt)
    return model