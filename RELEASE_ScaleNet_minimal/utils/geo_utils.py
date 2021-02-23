import torch

def fit_camH(bbox, H, v0, vc, f_pixels, y_person):
    # bbox: [x, y, w, h]
    # H: image height in px
    # input and return both in [top H, bottom 0] space
    vt = H - bbox[1]
    vb = H - (bbox[1] + bbox[3])
#     v0_single = yc * (vt - vb) / y_person + vb
    yc_single = y_person * (v0 - vb) / (vt - vb) / (1. + (vc - v0) * (vc - vt) / f_pixels**2)
    return yc_single

def fit_vt(yc_fit, vb, v0, vc, y_person_fit, inv_f2):
    # input and return both in [top H, bottom 0] space
    vt_camFit = (yc_fit * vb + y_person_fit * (v0 - vb) * (1. + inv_f2 * (vc - v0) * vc)) / (yc_fit + y_person_fit * (v0 - vb) * inv_f2 * (vc - v0))
    return vt_camFit

def horizon_from_pitch_vfov(pitch, vfov):
    v0 = 0.5 - 0.5*torch.tan(pitch) / torch.tan(vfov/2) # [up 0 bottom 1]
    return v0