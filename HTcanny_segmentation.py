import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def improved_canny(image):
    median = cv2.medianBlur(image, 3)

    gx = cv2.Sobel(median, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(median, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    high_thresh, _ = cv2.threshold(
        magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low_thresh = 0.5 * high_thresh

    edges = cv2.Canny(median, low_thresh, high_thresh)
    return edges


def generate_mask_from_edges(edges, original_img):
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(original_img)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".bmp")):
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            edges = improved_canny(img)
            mask = generate_mask_from_edges(edges, img)
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, mask)


if __name__ == "__main__":

    input_folder = "Input"     
    output_folder = "results/HTcanny"  
    process_folder(input_folder, output_folder)