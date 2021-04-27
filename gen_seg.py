import os
import cv2
import random
from copy import copy
import numpy as np
# for f in os.listdir('background'):
#     im = cv2.imread('background/'+f)
#     im = cv2.resize(im, (700, 700))
#     cv2.imwrite('background/'+f, im)

root = '/home/dung/Project/Product/cmt_data'


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


sizes = [(30, 30, 3), (30, 50, 3), (50, 30, 3), (50, 50, 3),
         (50, 50, 3), (100, 50, 3), (70, 50, 3), (70, 90, 3)]
bg = os.listdir('background')

noises = ['empty']
for size in sizes:
    noises.append(init_noise(size))


for f in os.listdir(root)[:1000]:
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
                new_w = 300
                new_h = int(h*new_w/w)
                body = cv2.resize(body, (new_w, new_h))
                for i in range(5):

                    noise = random.choice(noises)
                    n_body = np.zeros((new_h, new_w, 3))
                    n_body[:, :, :] = (255, 255, 255)
                    r_body = copy(body)

                    p_body = copy(body)
                    if noise != 'empty':
                        # shape of noise
                        nh, nw = noise.shape[:2]
                        # position start of noise
                        if i == 0:
                            sx, sy = 0, random.randint(0, new_h-nh)
                        elif i == 1:
                            sx, sy = random.randint(0, new_w-nw), 0
                        else:
                            sx, sy = random.randint(
                                0, new_w-nw), random.randint(0, new_h-nh)
                        if sx == 0 or sy == 0:
                            # body for rotate
                            r_body[sy:sy+nh, sx:sx+nw, :] = noise
                            # body for mask
                            n_body[sy:sy+nh, sx:sx+nw, :] = (0, 0, 0)
                    angle = random.randint(0, 359)
                    unet = rotate_bound(n_body, angle)
                    input = rotate_bound(r_body, angle, (random.randint(
                        0, 255), random.randint(0, 255), random.randint(0, 255)))

                    rotate_h, rotate_w = unet.shape[:2]
                    background_w, background_h = random.randint(
                        20, 50), random.randint(20, 50)
                    background_w, background_h = background_w + rotate_w, background_h+rotate_h
                    background = cv2.imread(
                        'background/{}'.format(random.choice(bg)))
                    bg_h, bg_w = background.shape[:2]
                    start_x, start_y = random.randint(
                        0, bg_w-background_w), random.randint(0, bg_h-background_h)

                    bg_random = background[start_y:start_y +
                                           background_h, start_x:start_x+background_w, :]

                    start_combine = (random.randint(
                        0, background_h-rotate_h), random.randint(0, background_w-rotate_w))
                    unet_last = np.zeros(
                        (bg_random.shape[0], bg_random.shape[1], 3))
                    input_last = copy(bg_random)
                    unet_last[start_combine[0]:start_combine[0]+rotate_h,
                              start_combine[1]:start_combine[1]+rotate_w, :] = unet
                    input_last[start_combine[0]:start_combine[0]+rotate_h,
                               start_combine[1]:start_combine[1]+rotate_w, :] = input
                    cv2.imwrite(
                        'data/input/{}_{}.png'.format(prefix, angle), input_last)
                    cv2.imwrite(
                        'data/unet/{}_{}.png'.format(prefix, angle), unet_last)
