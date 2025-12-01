

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2


class MFDNetEncoder:
    def __init__(self,
                 scale_f=2 ** 0.25,
                 min_scale=0.0,
                 max_scale=1,
                 min_size=256,
                 max_size=1024,
                 border=5,
                 reliability_thr=0.5,
                 repeatability_thr=0.4,
                 num_features=4096,
                 input_names=None,
                 output_names=None
                 ):
#        super().__init__('f')
        super().__init__()
        if input_names is None:
            self.input_names = ['img1_1', 'img1_2', 'img1_3', 'img1_4', 'img1_5', 'img1_6',
                                'img2_1', 'img2_2', 'img2_3', 'img2_4', 'img2_5', 'img2_6']
        if output_names is None:
            self.output_names = ['des1_1', 'des1_2', 'des1_3', 'des1_4', 'des1_5', 'des1_6',
                                 'des2_1', 'des2_2', 'des2_3', 'des2_4', 'des2_5', 'des2_6',
                                 'score1_1', 'score1_2', 'score1_3', 'score1_4', 'score1_5', 'score1_6',
                                 'score2_1', 'score2_2', 'score2_3', 'score2_4', 'score2_5', 'score2_6']
        self.scale_f = scale_f
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_size = min_size
        self.max_size = max_size
        self.border = border
        self.reliability_thr = reliability_thr
        self.repeatability_thr = repeatability_thr
        self.num_features = num_features
        cam_param = np.load(r'E:\Dataset\Defect-segmentation\cam_param.npz')
        self.new_mtxL, self.distL = (cam_param['ir_intrinsics'], cam_param['ir_distortion_coeffs'])
        self.homography_ir2vis, self.homography_vis2ir = cam_param['ir2vis'], cam_param['vis2ir']

    def encode(self, img_path1, img_path2):
        img1 = np.array(Image.open(img_path1).resize((8000, 6000), Image.Resampling.LANCZOS).convert('RGB'))
        img2 = np.array(Image.open(img_path2).convert('RGB'))
        # h, w = img2.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.new_mtxL, self.distL, (w, h), 1, (w, h))
        img2 = cv2.undistort(img2, self.new_mtxL, self.distL, None, self.new_mtxL)
        img1 = cv2.warpPerspective(img1, self.homography_vis2ir, (img2.shape[1], img2.shape[0]))
        # im_cb = self.checkboard(img1, img2)
        # im_cb = Image.fromarray(im_cb.astype(np.uint8))
        # im2_undist = Image.fromarray(img2.astype(np.uint8))
        # im_cb.save('vis_ir_encode.png')
        # im2_undist.save('ir_undist.png')

        img1 = TF.to_tensor(img1).unsqueeze(0)
        img1 = (img1 - img1.mean(dim=[-1, -2], keepdim=True)) / img1.std(dim=[-1, -2], keepdim=True)

        img2 = TF.to_tensor(img2).unsqueeze(0)
        img2 = (img2 - img2.mean(dim=[-1, -2], keepdim=True)) / img2.std(dim=[-1, -2], keepdim=True)
        img1_multiscale = self.generate_multiscale_img_list(img1, type='ndarray')
        img2_multiscale = self.generate_multiscale_img_list(img2, type='ndarray')

        input = img1_multiscale + img2_multiscale

        input_feed = dict(zip(self.input_names, input))
        return input_feed

    def decode(self, img_path1, img_path2, res):
        des1, des2, score1, score2 = res[0:6], res[6:12], res[12:18], res[18:24]
        bf = cv2.BFMatcher(crossCheck=True)
        img1 = np.array(Image.open(img_path1).resize((8000, 6000), Image.Resampling.LANCZOS).convert('RGB'))
        img2 = np.array(Image.open(img_path2).convert('RGB'))
        # h, w = img2.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.new_mtxL, self.distL, (w, h), 1, (w, h))
        img2 = cv2.undistort(img2, self.new_mtxL, self.distL, None, self.new_mtxL)
        img1 = cv2.warpPerspective(img1, self.homography_vis2ir, (img2.shape[1], img2.shape[0]))
        img1_array = img1.copy()
        img2_array = img2.copy()

        img1 = TF.to_tensor(img1).unsqueeze(0)
        img1 = (img1 - img1.mean(dim=[-1, -2], keepdim=True)) / img1.std(dim=[-1, -2], keepdim=True)

        img2 = TF.to_tensor(img2).unsqueeze(0)
        img2 = (img2 - img2.mean(dim=[-1, -2], keepdim=True)) / img2.std(dim=[-1, -2], keepdim=True)

        img1_multiscale = self.generate_multiscale_img_list(img1)
        img2_multiscale = self.generate_multiscale_img_list(img2)

        detector = NonMaxSuppression(
            rel_thr=self.reliability_thr,
            rep_thr=self.repeatability_thr)

        xys, desc, scores = self.extract_multiscale(img1, img1_multiscale, detector, des_input=des1,
                                                    repeat_input=score1)
        if len(scores) < self.num_features:
            idxs = scores.topk(len(scores))[1]
        else:
            idxs = scores.topk(self.num_features)[1]
        kp1 = xys[idxs].cpu().detach().numpy()
        desc1 = desc[idxs].cpu().detach().numpy()

        # extract keypoints/descriptors for a single image
        xys, desc, scores = self.extract_multiscale(img2, img2_multiscale, detector, des_input=des2,
                                                    repeat_input=score2)
        if len(scores) < self.num_features:
            idxs = scores.topk(len(scores))[1]
        else:
            idxs = scores.topk(self.num_features)[1]

        kp2 = xys[idxs].cpu().detach().numpy()
        desc2 = desc[idxs].cpu().detach().numpy()

        try:
            matches = bf.match(desc1, desc2)
        except:
            pass
        good = matches
        # matches = bf.knnMatch(desc1,desc2,k=2)
        # good = []
        # for m,n in matches:
        #     if m.distance < rt*n.distance:
        #         good.append(m)
        src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        src_im = cv2.applyColorMap(img2_array, cv2.COLORMAP_JET)[:, :, (2, 1, 0)]
        gt_im = img1_array
        if len(good) > 4:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=10.0, maxIters=100000)
            warpim = cv2.warpPerspective(gt_im, M, [gt_im.shape[1], gt_im.shape[0]])
            im_cb = self.checkboard(warpim, src_im)
            warpim = Image.fromarray(warpim.astype(np.uint8))
            im_cb = Image.fromarray(im_cb.astype(np.uint8))
        return M, im_cb, warpim

    def generate_multiscale_img_list(self, img, type='Tensor'):
        s = 1
        img_list = []
        if len(img.shape) == 4:
            B, three, H, W = img.shape
        else:
            three, H, W = img.shape
            B = 1
        assert B == 1 and three == 3
        while s + 0.001 >= max(self.min_scale, self.min_size / max(H, W)):
            if s - 0.001 <= min(self.max_scale, self.max_size / max(H, W)):
                if type == 'Tensor':
                    img_list.append(img)
                else:
                    img_list.append(img.cpu().numpy())
            s /= self.scale_f
            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
        return img_list

    def extract_multiscale(self, img, img_multi_scale, detector, des_input, repeat_input):
        old_bm = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # speedup

        # extract keypoints at multiple scales
        if len(img.shape) == 4:
            B, three, H, W = img.shape
        else:
            three, H, W = img.shape
            B = 1
        assert B == 1 and three == 3, "should be a batch with a single RGB image"

        s = 1.0  # current scale factor

        X, Y, S, C, Q, D = [], [], [], [], [], []

        for (img, desc, repeat) in zip(img_multi_scale, des_input, repeat_input):
            nh, nw = img.shape[2:]
            print(f"extracting at scale  = {nw:4d}x{nh:3d}")
            if type(desc) == 'torch.FloatTensor':
                descriptors = desc.detach().clone()
                repeatability = repeat.detach().clone()
            else:
                descriptors, repeatability = torch.from_numpy(desc), torch.from_numpy(repeat)

            mask = repeatability * 0
            mask[:, :, self.border:-self.border, self.border:-self.border] = 1
            repeatability = repeatability * mask
            y, x = detector(repeatability)  # nms
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]
            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            # S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            Q.append(q)
            D.append(d)

        # restore value
        torch.backends.cudnn.benchmark = old_bm

        Y = torch.cat(Y)
        X = torch.cat(X)
        # S = torch.cat(S) # scale
        scores = torch.cat(Q)  # scores = reliability * repeatability
        XYS = torch.stack([X, Y], dim=-1)
        D = torch.cat(D)
        return XYS, D, scores

    @staticmethod
    def checkboard(im1, im2, d=100):
        im1 = im1 * 1.0
        im2 = im2 * 1.0
        mask = np.zeros_like(im1)
        for i in range(mask.shape[0] // d + 1):
            for j in range(mask.shape[1] // d + 1):
                if (i + j) % 2 == 0:
                    mask[i * d:(i + 1) * d, j * d:(j + 1) * d, :] += 1
        return im1 * mask + im2 * (1 - mask)


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.6):
        super(NonMaxSuppression, self).__init__()
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rep_thr = rep_thr

    def forward(self, repeatability):
        # repeatability = repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        border_mask = maxima * 0
        border_mask[:, :, 10:-10, 10:-10] = 1
        maxima = maxima * border_mask
        print(maxima.sum())
        return maxima.nonzero().t()[2:4]


if __name__ == '__main__':
    import onnxruntime as rt
    import os
    import glob
    img_path_ir = glob.glob(r'E:\Dataset\Defect-segmentation\add_dataset\IR\*.png')
    sess = rt.InferenceSession("ust_image_registration_model.onnx")
    for ir_name in img_path_ir:
        encoder = MFDNetEncoder()
        file_name = os.path.splitext(os.path.basename(ir_name))[0]
        vis_path = os.path.dirname(os.path.dirname(ir_name))
        vis_name = '{}/VIS/{}.png'.format(vis_path, file_name)
        encode_input = encoder.encode(vis_name, ir_name)
        res = sess.run(encoder.output_names, encode_input)
        M, im_cb, warp_vis = encoder.decode(vis_name, ir_name, res)
        im_cb.save('{}/VIS-IR/{}.png'.format(vis_path, file_name))
        warp_vis.save('{}/VIS-ALIGN/{}.png'.format(vis_path, file_name))
