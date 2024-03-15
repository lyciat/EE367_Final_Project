import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from camera_resp import CameraRespFunct
from get_hdr_img import get_hdr, get_ldr
from tonemapping import my_tonemap

def df_to_numpy(df,key_val):
    n = df[key_val].to_numpy()
    return n[~np.isnan(n)]

def get_exposures(xlx_path, key_val):
    df = pd.read_excel(xlx_path)
    exp_db = df_to_numpy(df,key_val)
    return exp_db

def hdr_pipeline(key_val:str, xlx_path:str,path:str,path_rsp:str,key_val_rsp:str,l:int):
    exposures = get_exposures(xlx_path, key_val_rsp)
    crf = CameraRespFunct(path_rsp, l, exposures)
    print('Getting Camera Response')
    g_list, lE_list = crf.get_camera_resp()
    print('Getting back RAW Images from Camera Response')
    new_exposures = get_exposures(xlx_path, key_val)
    new_crf = CameraRespFunct(path,l, new_exposures)
    ldrs = get_ldr(crf, new_crf.images)
    # fig,axs=plt.subplots(1,len(ldrs))
    # for i, ld in enumerate(ldrs):
    #     axs[i].imshow(ld)
    # plt.show()
    print('Getting HDR Image')
    hdr = get_hdr(ldrs,exposures)
    # plt.imshow(hdr)
    # plt.title('HDR Image')
    # plt.axis('off')
    # plt.show()
    print('Apply Tonemapping')
    my_hdr, my_hdr_nobias = my_tonemap(hdr, gamma=0.8, saturation=0.4, bias=0.45)
    plt.imshow(my_hdr)
    plt.title('Our Tonemap with Bias')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    path = r"HooverTower/Hoover Tower"
    path_rsp = r"new_images"
    xlx_path = "Exposure_Times.xlsx"
    key_val = 'HooverTower'
    key_val_rsp = 'Munger'
    hdr_pipeline(key_val, xlx_path,path,path_rsp,key_val_rsp,1000)
    # lambdas = np.array([0.01, 0.1, 1, 10, 1000,10000])
    # exposures = [1/1, 1/5, 1/13, 1/25, 1/60, 1/100]
    # fig, axs = plt.subplots(2,3, figsize = (20, 10))
    # for i, l in enumerate(lambdas):
    #     crf = CameraRespFunct(path, l, exposures)
    #     g_list, lE_list = crf.getCameraResp()
    #     r = i // 3
    #     c = i % 3
    #     crf.plotResponseCurves(axs, r, c)
    # plt.show()