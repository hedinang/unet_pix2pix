import os
import cv2
import random
from copy import copy
import numpy as np
root = '/home/dung/Data/cmt_data'


def rotate_bound(image, angle, color=(0, 0, 0)):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), borderValue=color)


def init_noise(size):
    n = np.zeros(size)
    n[:, :, :] = (random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255))
    return n


sizes = [(100, 100, 3), (100, 200, 3), (200, 100, 3), (200, 200, 3),
         (200, 250, 3), (250, 200, 3), (100, 250, 3), (250, 100, 3),
         (250, 250, 3), (250, 150, 3), (150, 250, 3), (200, 150, 3), (150, 200, 3)]
noises = ['empty']
for size in sizes:
    noises.append(init_noise(size))


for f in os.listdir(root):
    if '.png' in f:
        img = cv2.imread('{}/{}'.format(root, f))
        h, w = img.shape[:2]
        prefix = f.split('.')[0]
        txt = open('{}/{}.txt'.format(root, prefix))
        lines = txt.read().split('\n')
        for line in lines[:-1]:
            cls, cx, cy, cw, ch = line.split(' ')
            cls, cx, cy, cw, ch = int(cls), float(
                cx), float(cy), float(cw), float(ch)
            if cls != 5:
                continue
            else:
                x1 = int((cx-cw/2)*w)
                y1 = int((cy-ch/2)*h)
                x2 = int((cx+cw/2)*w)
                y2 = int((cy+ch/2)*h)
                w, h = x2-x1, y2-y1
                body = img[y1:y2, x1:x2, :]
                for i in range(5):
                    noise = random.choice(noises)
                    n_body = np.zeros((h, w, 3))
                    n_body[:, :, :] = (255, 255, 255)
                    r_body = copy(body)
                    p_body = copy(body)
                    if noise != 'empty':
                        # shape of noise
                        nh, nw = noise.shape[:2]
                        # position start of noise
                        if i == 0:
                            sx, sy = 0, random.randint(0, h-nh)
                        elif i == 1:
                            sx, sy = random.randint(0, w-nw), 0
                        else:
                            sx, sy = random.randint(
                                0, w-nw), random.randint(0, h-nh)
                        # body for rotate
                        r_body[sy:sy+nh, sx:sx+nw, :] = noise
                        # body for pix2pix
                        p_body[sy:sy+nh, sx:sx+nw, :] = (0, 0, 0)
                        # body for mask
                        n_body[sy:sy+nh, sx:sx+nw, :] = (0, 0, 0)
                    angle = random.randint(0, 359)
                    unet = rotate_bound(n_body, angle)
                    input = rotate_bound(r_body, angle, (random.randint(
                        0, 255), random.randint(0, 255), random.randint(0, 255)))
                    pix2pix = rotate_bound(p_body, angle)
                    pix2pix = rotate_bound(pix2pix, -angle)
                    cv2.imwrite(
                        'data/input/{}_{}.png'.format(prefix, angle), input)
                    cv2.imwrite(
                        'data/unet/{}_{}.png'.format(prefix, angle), unet)
                    cv2.imwrite(
                        'data/pix2pix/{}_{}.png'.format(prefix, angle), pix2pix)
