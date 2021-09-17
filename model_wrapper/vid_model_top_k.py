import torch
import torch.nn as nn


# return k predicted results
class I3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor([0.39608, 0.38182, 0.35067], dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        std = torch.tensor([0.15199, 0.14856, 0.15698], dtype=torch.float32,
                           device=vid.get_device())[None, None, :, None, None]
        vid_t.sub_(mean).div_(std)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        vid_t = vid.clone()
        vid_t = vid_t.cuda()
        with torch.no_grad():
            logits = self.model.forward(self.preprocess(vid_t))
        logits = logits.squeeze(dim=2)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


# K LRCN predicted results
class LRCN_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        vid_t = vid_t.cuda()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                           device=vid.get_device())[None, None, :, None,None]
        vid_t.sub_(mean).div_(std)
        vid_t = vid_t.permute(1, 0, 2, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            logits = self.model.forward(self.preprocess(vid))
        logits = torch.mean(logits, dim=1)
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)


# return K C3D predicted results
class C3D_K_Model():
    def __init__(self, model):
        self.k = 1
        self.model = model

    def set_k(self, k):
        self.k = k

    def preprocess(self, vid):
        vid_t = vid.clone()
        mean = torch.tensor([101.2198, 97.5751, 89.5303],dtype=torch.float32,
                            device=vid.get_device())[None, None, :, None, None]
        vid_t = vid_t * 255
        vid_t[vid_t > 255] = 255
        vid_t[vid_t < 0] = 0
        vid_t.sub_(mean)
        vid_t = vid_t.permute(0, 2, 1, 3, 4)
        return vid_t

    def get_top_k(self, vid, k):
        with torch.no_grad():
            logits = self.model(self.preprocess(vid))
        top_val, top_idx = torch.topk(nn.functional.softmax(logits, 1), k)
        return top_val, top_idx, logits

    def __call__(self, vid):
        return self.get_top_k(vid, self.k)