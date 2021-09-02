import os
import glob

import cv2
import imageio
import scipy.io
import neuralbody.utils.geometry as geometry
import numpy as np
import torch

# AMASS dataset.
# https://amass.is.tue.mpg.de/


class AMASSData:
    def __init__(self, file_path):
        data = np.load(file_path)
        poses = data['poses']
        trans = data['trans']
        beta = data['betas']

        self.R = poses[..., :3].astype(np.float32)
        self.T = trans.astype(np.float32)

        self.theta = poses.astype(np.float32)
        self.theta[..., :3] = 0
        self.beta = beta.astype(np.float32)

    def __len__(self):
        return self.theta.shape[0]

    def __getitem__(self, idx):
        return self.smpl(idx)

    def smpl(self, i_frame):
        h2w_R = torch.tensor(self.R[i_frame])
        h2w_T = torch.tensor(self.T[i_frame])
        h2w_R = geometry.quat2mat(geometry.rodrigues(h2w_R))
        h2w = torch.cat([h2w_R, h2w_T[:, None]], dim=-1)
        h2w = torch.cat([h2w,torch.tensor([[0,0,0,1]])],dim=0).unsqueeze(0)

        pose = torch.tensor(self.theta[i_frame])
        pose = pose.reshape(-1,3).unsqueeze(0)
        shape = torch.tensor(self.beta).unsqueeze(0)

        return h2w, pose, shape

# ZJU mocap data
# https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md
class MoCapData:
    def __init__(self, data_root):
        self.data_root = data_root
        annots = np.load(os.path.join(data_root, 'annots.npy'),
                         allow_pickle=True).item()

        self.camK = np.array(annots['cams']['K']).astype(np.float32)
        self.camR = np.array(annots['cams']['R']).astype(np.float32)
        self.camT = np.array(annots['cams']['T'])/1000.0
        self.camT = self.camT.astype(np.float32)
        self.n_cam = len(self.camK)
        self.n_frame = len(annots['ims'])

        self.imgs = [ims['ims'] for ims in annots['ims']]
        # load params
        self.R = []
        self.T = []
        self.beta = []
        self.theta = []

        for i in range(self.n_frame):
            dd = np.load(os.path.join(
                self.data_root, 'params/{}.npy'.format(i+1)), allow_pickle=True).item()
            self.R.append(np.array(dd['Rh'][0]))
            self.T.append(np.array(dd['Th'][0]))
            self.beta.append(np.array(dd['shapes'])[0])
            self.theta.append(np.reshape(dd['poses'], (24, 3)))
        self.R = np.array(self.R).astype(np.float32)
        self.T = np.array(self.T).astype(np.float32)
        self.beta = np.array(self.beta).astype(np.float32)
        self.theta = np.array(self.theta).astype(np.float32)

        print('{} frames with {} views loaded'.format(self.n_frame, self.n_cam))

    def camera(self, i_view, scale=1.0):
        invR = self.camR[i_view].T
        invT = np.matmul(invR, -self.camT[i_view])
        c2w = np.hstack([invR, invT])
        c2w = torch.tensor(c2w)

        camK = self.camK[i_view]*scale
        camK[2, 2] = 1.0
        camK = torch.tensor(camK)

        return camK, c2w

    def smpl(self, i_frame):
        h2w_R = torch.tensor(self.R[i_frame])
        h2w_T = torch.tensor(self.T[i_frame])
        h2w_R = geometry.quat2mat(geometry.rodrigues(h2w_R))
        h2w = torch.cat([h2w_R, h2w_T[:, None]], dim=-1)
        h2w = torch.cat([h2w,torch.tensor([[0,0,0,1]])],dim=0)[None,...]

        pose = torch.tensor(self.theta[i_frame])[None,...]
        shape = torch.tensor(self.beta[i_frame])[None,...]

        return h2w, pose, shape

    def image(self, i_frame, i_view, scale=-1):
        file_path = self.imgs[i_frame][i_view]
        img = imageio.imread(os.path.join(
            self.data_root, file_path)).astype(np.float32)/255.0
        if scale > 0:
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        return torch.tensor(img)

    def mask(self, i_frame, i_view, scale=-1):
        file_name = self.imgs[i_frame][i_view][:-4]
        mask = imageio.imread(os.path.join(
            self.data_root, 'mask', file_name+'.png'))
        mask = (mask != 0).astype(np.uint8)

        mask_cihp = imageio.imread(os.path.join(
            self.data_root, 'mask_cihp', file_name+'.png'))
        mask_cihp = (mask_cihp != 0).astype(np.uint8)
        mask = (mask | mask_cihp)
        if scale > 0:
            mask = cv2.resize(mask, dsize=None, fx=scale,
                              fy=scale, interpolation=cv2.INTER_NEAREST)
        return torch.tensor(mask)

    def __len__(self):
        return self.n_frame



# DoubleFusion
# https://arxiv.org/pdf/1804.06023.pdf
class DoubleFusion:
    def __init__(self, data_root):
        self.data_root = data_root
        data = scipy.io.loadmat(os.path.join(data_root, 'camera.mat'))
        _camK = np.eye(3)
        _camK[:2, :2] = np.diag(data['color_focal_length'][0])
        _camK[:2, 2] = data['color_center'][0]
        self.camK = _camK.astype(np.float32)
        self.c2w = data['c2w'].astype(np.float32)

        rgb_frames = glob.glob(os.path.join(data_root, 'rgb/*.png'))
        self.rgb_format = 'rgb_frame_{}.png'
        self.obj_format = 'smpl_frame_{}.obj'
        indices = []
        for fname in rgb_frames:
            fname = os.path.basename(fname)
            fname = os.path.splitext(fname)[0]
            idx = int(fname[10:])
            if os.path.isfile(
                    os.path.join(data_root, 'obj',
                                 self.obj_format.format(idx))):
                indices.append(idx)

        self.indices = sorted(indices)

    @staticmethod
    def load_obj(file_path):
        v = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            elem = line.split()
            if elem[0] == 'v':
                v.append(elem[1:])
        v = np.array(v).astype(np.float32)
        return v

    def camera(self, idx, scale=1.0):
        camK = self.camK*scale
        camK[2, 2] = 1.0
        return torch.tensor(camK), torch.tensor(self.c2w)

    def image(self, idx, scale=1.0):
        img_path = os.path.join(self.data_root, 'rgb',
                                self.rgb_format.format(self.indices[idx]))
        img = imageio.imread(img_path)
        img = img.astype(np.float32)/255.0

        if scale < 1.0:
            img = cv2.resize(img, dsize=None,
                             fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # the last channel is useless
        return torch.tensor(img[..., :3])

    def smpl_v(self, idx):

        obj_path = os.path.join(self.data_root, 'obj',
                                self.obj_format.format(self.indices[idx]))
        v = self.load_obj(obj_path)
        v = torch.tensor(v)

        return v

    def __len__(self):
        return len(self.indices)
