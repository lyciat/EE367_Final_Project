import numpy as np
import matplotlib.pyplot as plt


def get_ldr(crf,images_arr):
    rad_R = np.exp(crf.gR)
    rad_G = np.exp(crf.gG)
    rad_B = np.exp(crf.gB)

    rad_R /= np.max(rad_R)
    rad_G /= np.max(rad_G)
    rad_B /= np.max(rad_B)
    
    ldr = np.zeros_like(images_arr, dtype=np.float64)
    for i, n in enumerate(images_arr):
        ### CV2 is BGR 
        ldr[i,:,:,2] = rad_B[n[:,:,0]]
        ldr[i,:,:,1] = rad_G[n[:,:,1]]
        ldr[i,:,:,0] = rad_R[n[:,:,2]]
    return ldr

def get_hdr(ldr_imgs, exposures):
    """
    From retrieved RAW images, get an HDR image using the combined
    exposures from those images

    """
    # compute weights
    weights = np.zeros_like(ldr_imgs)

    for i in range(ldr_imgs.shape[0]):
        img = ldr_imgs[i,:,:,:]
        for c in range(3):
            img_c = img[:,:,c]
            weights[i,:,:,c] = np.exp(-4*(((img_c - 0.5)**2)/0.5**2))

    # fuse LDR images using weights, make sure to store your fused HDR using the name hdr
    # initialize HDR image with all zeros
    hdr = np.zeros_like(ldr_imgs[0], dtype=float)

    for i in range(ldr_imgs.shape[0]):
        hdr += weights[i,:,:,:]*(np.log(ldr_imgs[i,:,:,:]+np.finfo(np.float32).eps) - np.log(exposures[i]))

    # Normalize
    scale = np.sum(weights, axis = 0)
    hdr = np.exp(hdr / scale)
    hdr *= 0.8371896/np.mean(hdr)  # this makes the mean of the created HDR image match the reference image (totally optional)

    # convert to 32 bit floating point format, required for OpenCV
    hdr = np.float32(hdr)

    return hdr

    