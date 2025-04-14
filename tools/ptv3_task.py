import numpy as np
import torch
from pandas.core.dtypes.inference import is_iterator
from mmdet3d.models.backbones.PointTransformerV3 import PointTransformerV3,Point


num_feat = 4
model = PointTransformerV3(cls_mode=True, in_channels=num_feat, enc_patch_size=(512, 512, 512, 512, 512)).cuda()
patch_size = 16384
batch_size = 2
batch_vals = torch.arange(0, batch_size, step=1)
repeat_vals = torch.tensor([patch_size for i in range(batch_size)])
batch_vals = torch.repeat_interleave(batch_vals, repeat_vals).cuda()
feats = torch.rand((patch_size*batch_size, num_feat)).cuda()
sample_data = {"feat": feats, "batch": batch_vals,
               "coord": feats[:,:3].cuda(), "grid_size": 0.01}
# print(sample_data)
sample_dict = Point(sample_data)
# print(sample_dict)
# for k,v in sample_dict.items():
#     print(f"{k}")
#     if is_iterator(v):
#         for item in v:
#             print(f"{item}")
#     else:
#         print(v)
output=model(sample_data)
print(output['feat'].shape)