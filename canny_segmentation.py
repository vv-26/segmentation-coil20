import cv2
import numpy as np
import os

def generate_mask(img):

    blur = cv2.GaussianBlur(img, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 100)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask



if __name__ == "__main__":

    input_folder = "Input"
    output_folder = "results/canny"
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".bmp")):
    
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
            mask = generate_mask(img)
    
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, mask)
    
    print("Canny mask generation completed")
