import os

import neuralbody.utils.geometry as geometry
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SMPL(nn.Module):
    def __init__(self, data_root, gender='neutral', spec='', beta=None):
        super(SMPL, self).__init__()
        file_path = os.path.join(
            data_root, 'SMPL{}_{}.npy'.format(spec, gender))
        data = np.load(file_path, allow_pickle=True).item()
        mu_ = data['v_template']
        n_betas_ = data['shapedirs'].shape[-1]
        shapedirs_ = np.reshape(data['shapedirs'], [-1, n_betas_]).T
        if isinstance(data['J_regressor'], np.ndarray):
            joint_reg_ = data['J_regressor'].T
        else:
            joint_reg_ = data['J_regressor'].T.todense()

        kin_parents_ = data['kintree_table'][0].astype(np.int32)
        blendW_ = data['weights']
        f_ = torch.tensor(data['f'].astype(np.int32), dtype=torch.long)
        vbyf_ = geometry.vertex_by_face(f_)

        self.n_J = joint_reg_.shape[1]
        self.n_V = mu_.shape[0]

        mu_ = torch.tensor(mu_, dtype=torch.float32)
        J_regressor = torch.tensor(joint_reg_, dtype=torch.float32)
        shapedirs_ = torch.tensor(shapedirs_, dtype=torch.float32)
        if beta is not None:
            v_res = torch.matmul(beta, shapedirs_)
            v_res = v_res.reshape(-1, self.n_V, 3)[0]
            mu_ = mu_ + v_res
        J = torch.matmul(J_regressor.T, mu_)

        self.register_buffer('shapedirs', shapedirs_)
        self.register_buffer('J_regressor', J_regressor)
        self.register_buffer('kin_parents', torch.tensor(
            kin_parents_, dtype=torch.long))
        self.register_buffer('f', f_)
        self.register_buffer('vbyf', vbyf_.to_dense())
        self.register_buffer('blendW', torch.tensor(
            blendW_, dtype=torch.float32))
        self.register_buffer('J', J)
        self.register_parameter('v', Parameter(mu_))

    def forward(self, theta, beta=None, h2w=None, inverse=False):
        if beta is None:
            v_shaped = self.v.expand([theta.shape[0], self.n_V, 3])
            J = self.J.expand([theta.shape[0], self.n_J, 3])
        else:
            v_res = torch.matmul(beta, self.shapedirs)
            v_res = v_res.reshape(-1, self.n_V, 3)
            v_shaped = v_res + self.v
            Jx = torch.matmul(v_shaped[..., 0], self.J_regressor)
            Jy = torch.matmul(v_shaped[..., 1], self.J_regressor)
            Jz = torch.matmul(v_shaped[..., 2], self.J_regressor)
            J = torch.stack([Jx, Jy, Jz], dim=-1)

        v = geometry.lbs(v_shaped, self.blendW, theta,
                         J, self.kin_parents, inverse)
        if h2w is not None:
            h2w_R = h2w[..., :3, :3]
            h2w_T = h2w[..., :3, 3:]
            v = torch.bmm(h2w_R, v.transpose(-1, -2)) + h2w_T
            v = v.transpose(-1, -2)
        return v, v_shaped
