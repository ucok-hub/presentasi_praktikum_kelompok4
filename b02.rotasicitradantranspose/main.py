import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Corrected path with escape characters
image_path = r"C:\\Users\\user\\OneDrive - Office 365 Original\\Documents\\----- PENTING -----\\TUGAS-TUGAS\\PNJ SEMESTER 4\\PENGOLAHAN CITRA DIGITAL\\PRAKTIKUM\\CHAPTER B\\b02.rotasicitradantranspose\\peter.jpg"

# Check if file exists
if not os.path.exists(image_path):
    print(f"File does not exist: {image_path}")
else:
    # Load the image
    image = cv2.imread(image_path)

    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def transpose_image(image):
        transposed = cv2.transpose(image)
        return transposed

    # Perform rotations
    rotated_45 = rotate_image(image, 45)
    rotated_minus_45 = rotate_image(image, -45)
    rotated_90 = rotate_image(image, 90)
    rotated_minus_90 = rotate_image(image, -90)
    rotated_180 = rotate_image(image, 180)

    # Perform transpose
    transposed = transpose_image(image)

    # Display the results
    titles = ['Original Image', 'Rotated 45°', 'Rotated -45°', 'Rotated 90°', 'Rotated -90°', 'Rotated 180°', 'Transposed']
    images = [image, rotated_45, rotated_minus_45, rotated_90, rotated_minus_90, rotated_180, transposed]

    plt.figure(figsize=(20, 10))
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()