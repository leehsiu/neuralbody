import torch.nn as nn
import spconv
import torch.nn.functional as F
import torch
from . import embedder


class Network(nn.Module):
    def __init__(self, n_frame, voxel_size):
        super(Network, self).__init__()

        self.voxel_size = voxel_size
        self.c = nn.Embedding(6890, 16)
        self.xyzc_net = SparseConvNet()

        self.latent = nn.Embedding(n_frame, 128)

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(352, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

    def get_feature_volume(self, indices, volume_sh, batch_size=1):
        code = self.c(torch.arange(0, 6890).to(indices.device))
        xyzc = spconv.SparseConvTensor(code, indices, volume_sh, batch_size)
        feature_volume = self.xyzc_net(xyzc)
        return feature_volume


    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)[0, :, 0]
        return features

    def query_density(self,pts,feature_volume, volume_bounds):

        # interpolate features
        grid_coords = 2 * \
            (pts[..., [2, 1, 0]] - volume_bounds[0]) / \
            (volume_bounds[1]-volume_bounds[0]) - 1
        grid_coords = grid_coords[..., [2, 1, 0]]
        grid_coords = grid_coords[None, None, :]
        h = self.interpolate_features(grid_coords, feature_volume)

        h = h.transpose(0, 1)

        # calculate density
        h = self.actvn(self.fc_0(h))
        h = self.actvn(self.fc_1(h))
        h = self.actvn(self.fc_2(h))

        sigma = self.alpha_fc(h)
        sigma = sigma.transpose(0, 1)

    
        return sigma


    def forward(self, pts, viewdir, feature_volume, volume_bounds, latent_index=None):

        if latent_index is None:
            latent_index = torch.full(pts.shape[:-1], 1, device=pts.device)

        # interpolate features
        grid_coords = 2 * \
            (pts[..., [2, 1, 0]] - volume_bounds[0]) / \
            (volume_bounds[1]-volume_bounds[0]) - 1
        grid_coords = grid_coords[..., [2, 1, 0]]
        grid_coords = grid_coords[None, None, :]
        h = self.interpolate_features(grid_coords, feature_volume)

        h = h.transpose(0, 1)

        # calculate density
        h = self.actvn(self.fc_0(h))
        h = self.actvn(self.fc_1(h))
        h = self.actvn(self.fc_2(h))

        sigma = self.alpha_fc(h)
        sigma = sigma.transpose(0, 1)

        # calculate color
        h = self.feature_fc(h)

        latent = self.latent(latent_index)
        latent = latent.transpose(1, 2)
        h = torch.cat((h, latent), dim=1)
        h = self.latent_fc(h)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        light_pts = embedder.xyz_embedder(pts)
        light_pts = light_pts.transpose(1, 2)

        h = torch.cat((h, viewdir, light_pts), dim=1)

        h = self.actvn(self.view_fc(h))
        rgb = self.rgb_fc(h)
        rgb = rgb.transpose(0, 1)

        return sigma, rgb


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
