# Object Segmentation on COIL-20

## Project Overview

This project builds an object segmentation pipeline using classical image processing techniques on the COIL-20 dataset. Three methods are implemented and compared: Sobel, Standard Canny, and Improved Canny, without using any machine learning.

## Dataset

COIL-20 Processed Dataset - 20 objects, 72 angles each, 1,440 grayscale images of size 128×128 pixels.  
Download: [COIL-20 Dataset](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)


## Methods

Sobel: Gaussian Blur → Sobel Gradient → Threshold → Closing → Contour → Mask  
Canny: Gaussian Blur → Canny Edge Detection → Closing → Contour → Mask  
Improved Canny: Bilateral Filter → Adaptive Threshold → Improved Canny → Closing → Contour → Mask  


## Requirements

```bash
pip install opencv-python numpy matplotlib
```

## How to Run

```bash
python canny_segmentation.py
python sobel_segmentation.py
python improved_canny.py
python visualize_pipeline.py
```

## Results

**Sobel** — Edge Precision: Moderate, Mask Accuracy: Moderate, Overall: Acceptable  
**Standard Canny** — Edge Precision: High, Mask Accuracy: High, Overall: Good  
**Improved Canny** — Edge Precision: Very High, Mask Accuracy: Very High, Overall: Best

## References

- S. A. Nene, S. K. Nayar, H. Murase, *COIL-20*, Columbia University, 1996.
- OpenCV Documentation: https://docs.opencv.org
