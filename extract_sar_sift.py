from re import sub
import torch
import os
import cv2
import numpy as np
import sys
import scipy.io as scio
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import lib.build_scale as build_scale
import lib.find_scale_extreme as find_scale_extreme
import lib.calc_descriptors as calc_descriptors
import lib.match as match
import time


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Extract keypoints for a given image")
    parser.add_argument("--subsets", type=str, default='UST_Campus', help='UST_Campus, Tower_D, Facade_18')
    parser.add_argument("--num_features", type=int, default=4096, help='number of keypoints')
    parser.add_argument("--contrastThreshold", type=float, default=-10000, help='number of keypoints')
    parser.add_argument("--edgeThreshold", type=float, default=-10000, help='number of keypoints')
    args = parser.parse_args()
    sigma = 2  # initial layer scale
    ratio = 2 ** (1 / 3.)  # scale ratio
    Mmax = 8  # layer number
    d = 0.04
    d_SH_1 = 0.8  # Harros function threshold
    d_SH_2 = 0.8  # Harros function threshold
    distRatio = 0.9
    if args.subsets == '+':
        args.subsets = ['VIS_NIR', 'VIS_IR', 'VIS_SAR']
    else:
        args.subsets = [args.subsets]
    feature_name = 'SAR_SIFT'
    if not os.path.exists(os.path.join(SCRIPT_DIR, 'features')):
        os.mkdir(os.path.join(SCRIPT_DIR, 'features'))
    time1 = time.time()
    for subset in args.subsets:
        if not os.path.exists(os.path.join(SCRIPT_DIR, 'features', args.subsets[0])):
            os.mkdir(os.path.join(SCRIPT_DIR, 'features', args.subsets[0]))
        if not os.path.exists(os.path.join(SCRIPT_DIR, 'features', args.subsets[0], feature_name)):
            os.mkdir(os.path.join(SCRIPT_DIR, 'features', args.subsets[0], feature_name))
        type1 = 'VIS'
        type2 = 'IR'
        imgs = os.listdir(os.path.join(subset, 'test', type1))
        for k, img in enumerate(imgs):
            img1_dir = os.path.join(subset, 'test', type1, img)
            img2_dir = os.path.join(subset, 'test', type2, img)
            img1 = cv2.imread(img1_dir)
            img2 = cv2.imread(img2_dir)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray1 = gray1 / 255.
            gray2 = gray2 / 255.
            sar_harris_function_1, gradient_1, angle_1 = build_scale.build_scale(gray1, sigma, Mmax, ratio, d)
            sar_harris_function_2, gradient_2, angle_2 = build_scale.build_scale(gray2, sigma, Mmax, ratio, d)
            GR_key_array_1 = find_scale_extreme.find_scale_extreme(sar_harris_function_1, d_SH_1, sigma, ratio,
                                                                   gradient_1, angle_1)
            GR_key_array_2 = find_scale_extreme.find_scale_extreme(sar_harris_function_2, d_SH_2, sigma, ratio,
                                                                   gradient_2, angle_2)
            descriptors_1, locs_1 = calc_descriptors.calc_descriptors(gradient_1, angle_1, GR_key_array_1)
            descriptors_2, locs_2 = calc_descriptors.calc_descriptors(gradient_2, angle_2, GR_key_array_2)
            kp1, kp2, desc1, desc2 = match.delete_duplications(GR_key_array_1[:, 0:2], GR_key_array_2[:, 0:2],
                                                             descriptors_1, descriptors_2)
            scio.savemat(os.path.join('./features', subset, feature_name, img.replace('.png', '.features.mat')),
                         {'desc1': desc1,
                          'kp1': kp1,
                          'desc2': desc2,
                          'kp2': kp2})
    time2 = time.time()
    print('Time:{:.2f}, imgs:{}'.format((time2 - time1), len(imgs)))

