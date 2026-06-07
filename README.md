# Brain MRI Tumor Segmentation Using Hybrid Fusion Algorithm

## Overview

This project implements a hybrid brain tumor segmentation framework based on Magnetic Resonance Imaging (MRI) scans. The methodology combines Wavelet Denoising, Expectation-Maximization (EM) Segmentation, Histogram-Based Thresholding, and Object-Based Thresholding to accurately identify tumor regions in brain MRI images.

The proposed fusion algorithm integrates the outputs of multiple segmentation techniques using a pixel-wise maximum criterion, resulting in improved tumor boundary detection and enhanced segmentation performance.

## Features

* MRI image preprocessing using Wavelet Denoising
* Expectation-Maximization (EM) based segmentation
* Histogram-based thresholding
* Object-based thresholding
* Hybrid fusion segmentation algorithm
* Tumor region extraction and visualization
* Performance comparison between individual and fused methods

## Methodology

The processing pipeline consists of the following stages:

1. MRI Image Acquisition
2. Wavelet-Based Noise Removal
3. EM Segmentation
4. Histogram Thresholding
5. Object-Based Thresholding
6. Fusion of Segmentation Results
7. Tumor Extraction and Visualization

## Technologies Used

* Python
* OpenCV
* NumPy
* PyWavelets
* Scikit-Image
* Scikit-Learn
* Matplotlib

## Project Structure

```text
brain_tumor_segmentation.py
brain_mri.png
wavelet_denoised.png
em_segmentation.png
histogram_segmentation.png
object_segmentation.png
fused_tumor.png
README.md
```

## Installation

```bash
pip install opencv-python numpy pywavelets scikit-image scikit-learn matplotlib
```

## Usage

```bash
python brain_tumor_segmentation.py
```

The script processes the input MRI image and generates segmented outputs for each method along with the final fused tumor segmentation result.

## Expected Outputs

* Wavelet Denoised MRI Image
* EM Segmentation Result
* Histogram Thresholding Result
* Object-Based Thresholding Result
* Final Fused Tumor Segmentation

## Applications

* Brain Tumor Detection
* Medical Image Analysis
* Clinical Decision Support Systems
* Computer-Aided Diagnosis
* Biomedical Research

## Future Enhancements

* Deep Learning-Based Segmentation
* 3D Tumor Reconstruction
* Multi-Modal MRI Analysis
* Automated Tumor Classification
* Integration with Clinical Workflows

## License

This project is intended for educational and research purposes.
