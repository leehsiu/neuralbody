import tqdm
import torch
import numpy as np
import pickle
import imageio
import glob
from neuralbody.config import cfg
import neuralbody.utils.geometry as geometry
from neuralbody.models import Network
from neuralbody.models.smpl import SMPL
from neuralbody.dataset import AMASSData,MoCapData
from neuralbody.renderer import Renderer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def float2uint8(src):
    return (255*np.clip(src, 0, 1)).astype(np.uint8)


def load_camera():
    camK = torch.eye(3)
    with open('mocap/mc/cam.pkl', 'rb') as f:
        dd = pickle.load(f, encoding='latin1')
    camK[0, 0] = dd['fx']
    camK[1, 1] = dd['fy']
    camK[0, 2] = dd['cx']
    camK[1, 2] = dd['cy']

    c2w = torch.eye(4)

    # fit plane
    depth = imageio.imread('mocap/mc/sample_900.depth.png')
    depth = depth.astype(np.float32)/1000.
    depth = torch.tensor(depth)
    cam_H, cam_W = depth.shape[:2]

    rays = geometry.camera_rays(camK, cam_W, cam_H, c2w, False, True)
    pts = rays[..., :3] + rays[..., 3:]*depth[..., None]

    pts = pts[500:550, 600:650]
    pts = pts.reshape(-1, 3)
    p_center, p_normal = geometry.fit_plane_svd(pts)

    p_center = p_center[None, :]
    p_normal = p_normal[None, :]

    unit_z = torch.zeros_like(p_normal)
    unit_z[..., 2] = 1.
    rot = geometry.vrrot(unit_z, -p_normal)[0]
    xy_rot = (45+180) / 180 * np.pi
    p2w = torch.eye(4)
    p2w[0, 0] = np.cos(xy_rot)
    p2w[1, 1] = np.cos(xy_rot)
    p2w[0, 1] = -np.sin(xy_rot)
    p2w[1, 0] = np.sin(xy_rot)

    p2w[:3, :3] = rot @ p2w[:3, :3]
    p2w[:3, 3] = p_center

    grid_x = torch.linspace(-1, 1, 10)
    grid_y = torch.linspace(-1, 1, 10)
    grid_x, grid_y = torch.meshgrid(grid_x, grid_y)
    grid_xyz = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1)
    grid_xyz = grid_xyz.reshape(-1, 3)
    grid_xyz = grid_xyz @ p2w[:3, :3].T + p2w[:3, 3]

    return camK, c2w, p2w, grid_xyz


def main():

    scale = 0.5

    # 1. load model
    model = Network(cfg.num_train_frame, cfg.voxel_size)
    model = model.to(device)
    model_path = 'weights/ZJU_313.pth'
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['net'])


    train_data = MoCapData('data/zju_mocap/CoreView_313')
    _,_,beta = train_data.smpl(0)
    train_wrapper = SMPL('mocap/smpl', gender='neutral',
                         spec='', beta=beta)
    # 2. prepare dataset

    renderer = Renderer(model, cfg.N_samples, cfg.perturb, device)

    camK, c2w, p2w, _ = load_camera()
    camK = camK*scale
    camK[2, 2] = 1.

    mosh_files = glob.glob('mocap/mosh/*.npz')
    mosh_files.sort()

    for i in [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18]:
        amass = AMASSData(mosh_files[i])

        # align h2w0 to p2w
        # /**===============================================
        h2w0, _, beta = amass.smpl(0)
        h2w0 = h2w0[0]
        h2w0[:3, :3] = torch.eye(3)

        local_wrapper = SMPL('mocap/smpl', gender='male',
                             SMPLH=True, beta=beta)

		# copy training template mesh here
        local_wrapper.v = train_wrapper.v
        v_min = torch.min(local_wrapper.v, dim=0)[0]
        z_offset = torch.eye(4)
        z_offset[2, 3] = - v_min[1]
        h2w_offset = p2w  @ z_offset @ torch.inverse(h2w0)
        #  ===============================================**/

        img_mats = []
        for i_frame in range(0, len(amass), 12):
            h2w, pose, _ = amass.smpl(i_frame)
            h2w = h2w_offset @ h2w[0]
            c2h = torch.inverse(h2w) @ c2w

            with torch.no_grad():
                keypoints = local_wrapper.forward(pose, None)[0]
                keypoints = keypoints[0]
                ret = renderer(camK, c2h, keypoints)

            rgb = ret['rgb'].cpu().numpy()
            alpha = ret['alpha'].cpu().numpy()
            rgb = float2uint8(rgb)
            alpha = float2uint8(alpha)
            img_mats.append(rgb)


            print('{}:{}/{}'.format(i, i_frame, len(amass)))
            imageio.imwrite('render/rgb-{}-{}.png'.format(i, i_frame), rgb)
            imageio.imwrite('render/alpha-{}-{}.png'.format(i, i_frame), alpha)
        imageio.mimwrite('render/render-{}.gif'.format(i), img_mats)


if __name__ == '__main__':
    main()
