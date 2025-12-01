
import os
import shutil
from PIL import Image
import cv2
import scipy.io as scio
import numpy as np
from matplotlib.image import imread


if not os.path.exists('./test'):
    os.mkdir('./test')

else:
    shutil.rmtree('./test')
    os.mkdir('./test')
    
if not os.path.exists('./train'):
    os.mkdir('./train')

else:
    shutil.rmtree('./train')
    os.mkdir('./train')
    
os.mkdir('./test/VIS')
os.mkdir('./test/IR')
os.mkdir('./train/VIS')
os.mkdir('./train/IR')

shutil.unpack_archive('./transforms.zip', './')
shutil.unpack_archive('./landmarks.zip', './test')
test_list = open('./test.txt','r').read().split('\n')
train_list = open('./train.txt','r').read().split('\n')

for i,train_img in enumerate(train_list):
    if len(train_img)>2:
        if train_img.startswith('lwir'):
            try:
                vis_img = Image.open('./lghd_icip2015_rgb_lwir/rgb/'+train_img.replace('png','bmp').replace('lwir','rgb'))
                IR_img = imread('./lghd_icip2015_rgb_lwir/lwir/'+train_img)
                IR_convert = (IR_img*255).astype(np.uint8)
                IR_img = Image.fromarray(IR_convert)
            except:
                continue
        elif train_img.startswith('FLIR'):
            vis_img = Image.open('./RoadScene/crop_LR_visible/'+train_img)
            IR_img = Image.open('./RoadScene/cropinfrared/'+train_img)
        vis_img.save('./train/VIS/{}.png'.format(i+1))
        IR_img.save('./train/IR/{}.png'.format(i+1))

for i,test_img in enumerate(test_list):
    if len(test_img)>2:
        if test_img.startswith('lwir'):
            try:
                vis_img = Image.open('./lghd_icip2015_rgb_lwir/rgb/'+test_img.replace('png','bmp').replace('lwir','rgb'))
                IR_img = imread('./lghd_icip2015_rgb_lwir/lwir/'+train_img)
                IR_convert = (IR_img*255).astype(np.uint8)
                IR_img = Image.fromarray(IR_convert)
            except:
                continue
        elif test_img.startswith('FLIR'):
            vis_img = Image.open('./RoadScene/crop_LR_visible/'+test_img)
            IR_img = Image.open('./RoadScene/cropinfrared/'+test_img)
            
        if os.path.exists('./test/transforms/{}.21.mat'.format(i+1)):
            H = scio.loadmat('./test/transforms/{}.21.mat'.format(i+1))['H']
            IR_img = np.array(IR_img)
            IR_img = cv2.warpPerspective(IR_img,H,[IR_img.shape[1],IR_img.shape[0]])
            IR_img = Image.fromarray(IR_img)
        if os.path.exists('./test/transforms/{}.12.mat'.format(i+1)):
            H = scio.loadmat('./test/transforms/{}.12.mat'.format(i+1))['H']
            vis_img = np.array(vis_img)
            vis_img = cv2.warpPerspective(vis_img,H,[vis_img.shape[1],vis_img.shape[0]])
            vis_img = Image.fromarray(vis_img)
        IR_img.save('./test/IR/{}.png'.format(i+1))
        vis_img.save('./test/VIS/{}.png'.format(i+1))