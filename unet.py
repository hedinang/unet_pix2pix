from unet_model import UNet2
import torch
from torch import nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs')


class DocSegDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.ids = os.listdir('{}/input'.format(root))
        self.transform_color = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        color = Image.open('{}/input/{}'.format(self.root, self.ids[idx]))
        color = self.transform_color(color)
        gray = Image.open('{}/unet/{}'.format(self.root,
                                              self.ids[idx])).convert('L')
        gray = self.transform_gray(gray)
        return color, gray, '{}/input/{}'.format(self.root, self.ids[idx])


device = torch.device('cuda')
model = UNet2(3, 1)
model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dataset = DocSegDataset('data')
data = DataLoader(dataset, batch_size=1, shuffle=True)
epoch = 1000
step = 0
for i in tqdm(range(epoch)):
    print('start epoch: ', i)
    for color, gray, path in data:
        step += 1
        color = color.to(device)
        gray = gray.to(device)
        output = model(color)
        loss = criterion(output, gray)
        if step % 1 == 0:
            print('loss: ', loss)
            writer.add_scalar('loss', loss, step)
            c = cv2.imread(path[0])
            h, w = c.shape[:2]
            new_w = 400
            new_h = int(400 * h/w)
            c = cv2.resize(c, (new_w, new_h))
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            c = torch.tensor(c, dtype=torch.uint8).permute(2, 0, 1)
            g = cv2.imread(path[0].replace('input', 'unet'))
            g = cv2.resize(g, (new_w, new_h))
            g = cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
            g = torch.tensor(g, dtype=torch.uint8).permute(2, 0, 1)
            o = (output[0] > 0.5).float()
            _, h, w = o.shape
            refer = np.zeros((h, w, 3))
            for ii in range(h):
                for ij in range(w):
                    if o[:, ii, ij] == 1:
                        refer[ii, ij, :] = (255, 255, 255)
            refer = np.resize(refer, (new_h, new_w, 3))
            refer = torch.tensor(refer, dtype=torch.uint8).permute(2, 0, 1)
            combine = torch.cat([c, g, refer], dim=2)

            writer.add_image('image', combine, step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), '2.pth')
