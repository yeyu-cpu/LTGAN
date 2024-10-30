import os
import cv2
from PIL import Image
#
# 打开原始图像
def resize_image(file_name):
    image = Image.open(f'GeomConsistentFR/FFHQ_skin_masks/{file_name}')
    resized_image = image.resize((256, 256))
    resized_image.save(f'GeomConsistentFR/FFHQ_skin_masks/{file_name}')
    image = cv2.imread(f'GeomConsistentFR/FFHQ_skin_masks/{file_name}')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f'GeomConsistentFR/FFHQ_skin_masks/{file_name}', binary_image)
    return binary_image
