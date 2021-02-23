import torch
import numpy as np


def generate_field(f, xi, H, W, el, roll, device):
    u0 = W / 2.
    v0 = H / 2.

    grid_x, grid_y = torch.meshgrid(torch.arange(0, W).float().to(device), torch.arange(0, H).float().to(device))

    x_ref = 1
    y_ref = 1

    X_Cam = (grid_x - u0) / f
    Y_Cam = -(grid_y - v0) / f

    #from matplotlib import pyplot as plt
    #plt.subplot(121); plt.imshow(X_Cam); plt.colorbar()
    #plt.subplot(122); plt.imshow(Y_Cam); plt.colorbar()
    #plt.show()
    #import pdb; pdb.set_trace()

    # 2. Projection on the sphere

    AuxVal = X_Cam*X_Cam + Y_Cam*Y_Cam

    #alpha_cam = np.real(xi + torch.sqrt(1 + (1 - xi*xi)*AuxVal))
    alpha_cam = xi + torch.sqrt(1 + (1 - xi*xi)*AuxVal)

    alpha_div = AuxVal + 1

    alpha_cam_div = alpha_cam / alpha_div

    X_Sph = X_Cam * alpha_cam_div
    Y_Sph = Y_Cam * alpha_cam_div
    Z_Sph = alpha_cam_div - xi
    #from matplotlib import pyplot as plt
    #import pdb; pdb.set_trace()

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.subplot(121); plt.imshow(X_Cam); plt.colorbar()
    #plt.subplot(122); plt.imshow(Y_Cam); plt.colorbar()

    #plt.figure()
    #plt.subplot(131); plt.imshow(X_Sph); plt.colorbar()
    #plt.subplot(132); plt.imshow(Y_Sph); plt.colorbar()
    #plt.subplot(133); plt.imshow(Z_Sph); plt.colorbar()

    # 3. Rotation of the sphere

    #idx1 = np.array([[0], [0], [0]])
    #idx2 = np.array([[1], [1], [1]])
    #idx3 = np.array([[2], [2], [2]])
    #elems1 = rot[:, 0]
    #elems2 = rot[:, 1]
    #elems3 = rot[:, 2]

    #x1 = elems1[0] * X_Sph + elems2[0] * Y_Sph + elems3[0] * Z_Sph
    #y1 = elems1[1] * X_Sph + elems2[1] * Y_Sph + elems3[1] * Z_Sph
    #z1 = elems1[2] * X_Sph + elems2[2] * Y_Sph + elems3[2] * Z_Sph
    #import pdb; pdb.set_trace()
    #coords = torch.stack((X_Sph.reshape(-1), Y_Sph.reshape(-1), Z_Sph.reshape(-1)))
    # Gradient not passing
    #rot_el = torch.Tensor([[1., 0., 0.], [0., torch.cos(el), -torch.sin(el)], [0., torch.sin(el), torch.cos(el)]]).float().to(device)
    #rot_az = torch.Tensor([[torch.cos(az), 0., torch.sin(az)], [0., 1., 0.], [-torch.sin(az), 0., torch.cos(az)]])
    #rot_roll = torch.Tensor([[torch.cos(roll), -torch.sin(roll), 0.], [torch.sin(roll), torch.cos(roll), 0.], [0., 0., 1.]]).float().to(device)
    #sph = rot_roll.transpose(1, 0).matmul(rot_el.matmul(coords))

    # rot_el
    cosel, sinel = torch.cos(el), torch.sin(el)
    Y_Sph = Y_Sph*cosel - Z_Sph*sinel
    Z_Sph = Y_Sph*sinel + Z_Sph*cosel

    #rot_roll
    cosroll, sinroll = torch.cos(roll), torch.sin(roll)
    X_Sph = X_Sph*cosroll - Y_Sph*sinroll
    Y_Sph = X_Sph*sinroll + Y_Sph*cosroll

    #sph = rot_az.dot(sph)

    #sph = sph.reshape((3, H, W))#.transpose((1,2,0))
    coords = torch.stack((X_Sph.reshape(-1), Y_Sph.reshape(-1), Z_Sph.reshape(-1)))
    coords = coords / torch.sqrt(torch.sum(coords**2, dim=0))
    return coords
    #import pdb; pdb.set_trace()
    #X_Sph, Y_Sph, Z_Sph = sph[0,:,:], sph[1,:,:], sph[2,:,:]

    #X_Sph = x1
    #Y_Sph = y1
    #Z_Sph = z1

    #from matplotlib import pyplot as plt
    #plt.figure()
    #plt.subplot(131); plt.imshow(X_Sph); plt.colorbar()
    #plt.subplot(132); plt.imshow(Y_Sph); plt.colorbar()
    #plt.subplot(133); plt.imshow(Z_Sph); plt.colorbar()
    #plt.show()

    # 4. cart 2 sph
    ntheta = torch.atan2(X_Sph, Z_Sph)
    nphi = torch.atan2(Y_Sph, torch.sqrt(Z_Sph**2 + X_Sph**2))

    return ntheta, nphi

    pi = m.pi

    # 5. Sphere to pano
    min_theta = -pi
    max_theta = pi
    min_phi = -pi / 2.
    max_phi = pi / 2.

    min_x = 0
    max_x = ImPano_W - 1.0
    min_y = 0
    max_y = ImPano_H - 1.0

    ## for x
    a = (max_theta - min_theta)
    b = max_theta - a * max_x  # from y=ax+b %% -a;
    nx = (1. / a)* (ntheta - b)

    ## for y
    a = (min_phi - max_phi)
    b = max_phi - a * min_y  # from y=ax+b %% -a;
    ny = (1. / a)* (nphi - b)
