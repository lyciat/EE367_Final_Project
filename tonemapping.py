import cv2
import matplotlib.pyplot as plt
import numpy as np
from get_hdr_img import get_hdr

def show_drago(hdr_img, gamma=1, saturation=0.7, bias=0.85):
    # Tonemap using Drago's method to obtain 24-bit color image
    tonemapDrago = cv2.createTonemapDrago(gamma,saturation,bias)
    ldrDrago = tonemapDrago.process(hdr_img)
    plt.title("Drago Tone Mapping", fontsize=20)
    plt.axis('off')
    plt.imshow(ldrDrago)

def show_mantiuk(hdr_img):
    gamma_mantiuk = 2.2
    scale_mantiuk = 0.85
    sat_mantiuk = 1.2

    tonemapMantiuk = cv2.createTonemapMantiuk(gamma_mantiuk, scale_mantiuk, sat_mantiuk)
    ldrMantiuk =  tonemapMantiuk.process(hdr_img)
    ldrMantiuk = 3 * ldrMantiuk
    plt.imshow(ldrMantiuk)
    plt.title('Mantiuk Tone Map')
    plt.axis('off')


def my_tonemap(hdr, gamma=0.8, saturation=0.4, bias=0.45):

    bias_term = np.log(bias) / np.log(.5)

    my_hdr = (saturation * (hdr)**bias_term)**gamma
    my_hdr = np.uint8(np.clip(my_hdr, 0, 1) *255)

    my_hdr_simple = (saturation * hdr)**gamma
    my_hdr_simple = np.uint8(np.clip(my_hdr_simple, 0, 1) *255)
    return my_hdr, my_hdr_simple