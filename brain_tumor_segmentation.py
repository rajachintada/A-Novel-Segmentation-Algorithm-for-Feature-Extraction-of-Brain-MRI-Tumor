
import cv2
import numpy as np
import pywt
from sklearn.mixture import GaussianMixture
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

def wavelet_denoise(image):
    coeffs = pywt.wavedec2(image, 'db4', level=2)
    threshold = np.median(np.abs(coeffs[-1][0])) / 0.6745
    threshold *= np.sqrt(2 * np.log(image.size))

    denoised_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        denoised_level = tuple(
            pywt.threshold(c, threshold, mode='soft')
            for c in detail_level
        )
        denoised_coeffs.append(denoised_level)

    denoised = pywt.waverec2(denoised_coeffs, 'db4')
    denoised = np.clip(denoised, 0, 255)
    return denoised.astype(np.uint8)

def em_segmentation(image):
    pixels = image.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type='full',
        random_state=42
    )

    gmm.fit(pixels)
    labels = gmm.predict(pixels)

    means = gmm.means_.flatten()
    tumor_cluster = np.argmax(means)

    segmented = (labels == tumor_cluster)
    return segmented.reshape(image.shape).astype(np.uint8)

def histogram_thresholding(image):
    thresh = threshold_otsu(image)
    return (image > thresh).astype(np.uint8)

def object_based_thresholding(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    binary = remove_small_objects(binary, min_size=100)
    return binary.astype(np.uint8)

def fusion_segmentation(em_mask, hist_mask, obj_mask):
    fused = np.maximum(em_mask, hist_mask)
    fused = np.maximum(fused, obj_mask)
    return fused.astype(np.uint8)

if __name__ == "__main__":
    image = cv2.imread("brain_mri.png", cv2.IMREAD_GRAYSCALE)

    denoised = wavelet_denoise(image)
    em_result = em_segmentation(denoised)
    hist_result = histogram_thresholding(denoised)
    obj_result = object_based_thresholding(denoised)

    fused_tumor = fusion_segmentation(
        em_result,
        hist_result,
        obj_result
    )

    cv2.imwrite("wavelet_denoised.png", denoised)
    cv2.imwrite("em_segmentation.png", em_result * 255)
    cv2.imwrite("histogram_segmentation.png", hist_result * 255)
    cv2.imwrite("object_segmentation.png", obj_result * 255)
    cv2.imwrite("fused_tumor.png", fused_tumor * 255)

    print("Tumor segmentation completed.")
