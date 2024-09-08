# from PIL import Image

# def merge_images(file1, file2):
#     """Merge two images into one, displayed side by side
#     :param file1: path to first image file
#     :param file2: path to second image file
#     :return: the merged Image object
#     """
#     image1 = Image.open(file1)
#     image2 = Image.open(file2)

#     (width1, height1) = image1.size
#     (width2, height2) = image2.size

#     result_width = width1 + width2
#     result_height = max(height1, height2)

#     result = Image.new('RGB', (result_width, result_height))
#     result.paste(im=image1, box=(0, 0))
#     result.paste(im=image2, box=(width1, 0))
#     return result

from cv2 import vconcat
import numpy as np

def stitch(image1, image2):
    """give the image1 and image2 as paths of the actual files"""
    img1 = cv2.imread(image1) 
    img2 = cv2.imread(image2)
    im_v = cv2.vconcat([img1, img2]) 
    return im_v

def detect_document(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur (need to fix this)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # There's a better algorithm to find this
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            return approx

    return None


def apply_gaussian_blur(image, rois):
    for roi in rois:
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        blurred_roi = cv2.GaussianBlur(roi_image, (15, 15), 0)
        # Replacement with blurred image
        image[y:y+h, x:x+w] = blurred_roi
    return image
