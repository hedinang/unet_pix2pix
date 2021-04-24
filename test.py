from torchsummary import summary
import torch
from networks import *
device = torch.device('cuda')
model = LocalEnhancer(3, 3).to(device)
print(model)
finetune_list = set()
params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.startswith('model' + str(2)):
        params += [value]
        finetune_list.add(key.split('.')[0])