import torch
import torch.nn.functional as F
import neuralbody.utils.geometry as geometry
from neuralbody.utils.batchify import batchify


class Renderer:

    def __init__(self, model, N_samples, stratified, device):
        self.model = model
        self.N_samples = N_samples
        self.stratified = stratified
        self.device = device

    def sigma2alpha(self, sigma, dists):
        return 1.-torch.exp(-sigma*dists)

    def sigma2weights(self, sigma, dists):
        alpha = self.sigma2alpha(sigma, dists)
        weights = alpha \
            * torch.cumprod(
                torch.cat([torch.ones([alpha.shape[0], 1], device=sigma.device),
                           1.-alpha + 1e-10], -1),
                dim=-1)[:, :-1]
        return weights

    def depth_sampling(self, rays, bbox, near=0., far=1., eps=1e-7):
        N_rays = rays.shape[0]

        if bbox is not None:
            bounds = geometry.ray_cube_intersection(rays, bbox)
            z_min = bounds[..., 0]
            z_max = bounds[..., 1]
            z_min[z_min < near] = near
        else:
            z_min = near
            z_max = far

        valid = torch.where((z_max-z_min) > eps)[0]

        if len(valid) == 0:
            return None, valid

        t_vals = torch.linspace(
            0., 1., steps=self.N_samples, device=self.device)
        t_vals = t_vals.expand([N_rays, self.N_samples])

        if self.stratified:
            mids = .5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], -1)
            lower = torch.cat([t_vals[..., :1], mids], -1)
            t_rand = torch.rand(t_vals.shape).to(self.device)
            t_vals = lower + (upper - lower) * t_rand

        z_vals = z_min[..., None]*(1.-t_vals) + z_max[..., None]*t_vals

        return z_vals, valid

    def prepare_spconv_input(self, xyz):

        min_xyz = torch.min(xyz, dim=0)[0]
        max_xyz = torch.max(xyz, dim=0)[0]


        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05

        dhw = xyz[:, [2, 1, 0]]
        min_dhw = min_xyz[[2, 1, 0]]
        max_dhw = max_xyz[[2, 1, 0]]

        voxel_size = torch.tensor(self.model.voxel_size)

        indices = torch.round((dhw-min_dhw)/voxel_size).int()
        volume_sh = torch.ceil((max_dhw - min_dhw) / voxel_size).int()
        size_divisible = 32
        volume_sh = (volume_sh | (size_divisible - 1)) + 1
        max_dhw = volume_sh * voxel_size + min_dhw
        volume_bounds = torch.stack([min_dhw, max_dhw], dim=0).to(self.device)

        volume_sh = volume_sh.to(self.device)
        volume_sh = volume_sh.tolist()

        indices = torch.cat(
            [torch.full([indices.shape[0], 1], 0, dtype=indices.dtype), indices], dim=1)
        indices = indices.to(self.device)

        return indices, volume_sh, volume_bounds

    def render_rays(self, rays, volume_bounds, feature_volume):
        N_rays = rays.shape[0]
        bbox = volume_bounds[:, [2, 1, 0]]
        bbox = geometry.bbox_3d(bbox[0], bbox[1])

        # sampling points along camera rays
        z_vals, valid = self.depth_sampling(rays, bbox)

        outputs = {}
        outputs['rgb'] = torch.zeros((N_rays, 3), device=self.device)
        outputs['alpha'] = torch.zeros((N_rays, 1), device=self.device)

        if len(valid) == 0:
            return outputs

        rays_o, rays_d = rays[..., :3], rays[..., 3:6]
        z_valid = z_vals[valid]
        o_valid = rays_o[valid]
        d_valid = rays_d[valid]

        pts = o_valid[..., None, :] + \
            d_valid[..., None, :] * z_valid[..., :, None]
        viewdir = F.normalize(d_valid, dim=-1)
        viewdir = viewdir[:, None, :].expand_as(pts)

        sigma, rgb = batchify(self.model, 1024*64, pts, viewdir,
                              feature_volume=feature_volume, volume_bounds=volume_bounds)

        sigma = F.relu(sigma[0, ...])
        rgb = torch.sigmoid(rgb)

        dists = z_valid[:, 1:] - z_valid[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10], device=self.device).expand(
            dists[..., :1].shape)], -1)
        dists = dists * torch.norm(d_valid[..., None, :], dim=-1)

        # Pretrained NeRF uses F.relu
        weights = self.sigma2weights(sigma, dists)
        rgb_map = torch.sum(weights[None, ...]*rgb, dim=-1)

        rgb_map = rgb_map.transpose(0, 1)
        acc_map = torch.sum(weights, dim=-1)

        # volumetric rendering
        outputs['rgb'][valid] = rgb_map
        outputs['alpha'][valid] = acc_map[..., None]

        return outputs

    def __call__(self, camK, c2w, xyz, rays=None, chunk=1024, img_W=None, img_H=None):

        if rays is None:
            rays = geometry.camera_rays(camK, img_W, img_H, c2w, False, True)
        rays = rays.to(self.device)
        sh = rays.shape
        rays_flat = rays.reshape(-1, 6)

        # encode neural body
        indices, volume_sh, volume_bounds = self.prepare_spconv_input(xyz)

        feature_volume = self.model.get_feature_volume(indices, volume_sh)

        all_ret = batchify(self.render_rays, chunk, rays_flat, volume_bounds=volume_bounds,
                           feature_volume=feature_volume)

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        return all_ret
