from skimage import metrics as measure
import numpy as np 


def compute_psnr(generated_image, original_image, maskit=False):
    data1 = np.asarray( generated_image, dtype="float32" )    
    max1 = np.max(data1)
    data1 /= max1

    data2 = np.asarray( original_image, dtype="float32" )
    max2 = np.max(data2)
    data2 /= max2
    if maskit:
        mask = np.zeros(data1.shape)
        mask[data2>0] = data1[data2>0]
        data1 = mask
    psnr = measure.peak_signal_noise_ratio(data2,data1, data_range=1.0)
    return psnr

def compute_ssim(generated_image,original_image, maskit=False):
    data1 = np.asarray( generated_image, dtype="float32" )    
    max1 = np.max(data1)
    data1 /= max1

    data2 = np.asarray( original_image, dtype="float32" )    
    max2 = np.max(data2)
    data2 /= max2
    if maskit:
        mask = np.zeros(data1.shape)
        mask[data2>0] = data1[data2>0]
        data1 = mask
    ssim = measure.structural_similarity(data2,data1,  sigma=1.5, gaussian_weights=True, use_sample_covariance=False, data_range=1.0)
    return ssim                        