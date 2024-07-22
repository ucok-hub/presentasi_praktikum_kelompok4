import cv2
import matplotlib.pyplot as plt

# Corrected image path
image_path = r"C:\Users\user\OneDrive - Office 365 Original\Documents\----- PENTING -----\TUGAS-TUGAS\PNJ SEMESTER 4\PENGOLAHAN CITRA DIGITAL\PRAKTIKUM\CHAPTER B\b03.resize\peter.jpg"

# Load image from the specified directory
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found at the specified path.")
else:
    # Scaling factors
    scale_up_factor = 2
    scale_down_factor = 0.5
    new_dimensions = (900, 400)

    # Resize image - Zoom In
    resized_zoom_in = cv2.resize(image, None, fx=scale_up_factor, fy=scale_up_factor, interpolation=cv2.INTER_LINEAR)

    # Resize image - Zoom Out
    resized_zoom_out = cv2.resize(image, None, fx=scale_down_factor, fy=scale_down_factor, interpolation=cv2.INTER_LINEAR)

    # Resize image to specific dimensions
    resized_specific = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB for displaying using matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_zoom_in_rgb = cv2.cvtColor(resized_zoom_in, cv2.COLOR_BGR2RGB)
    resized_zoom_out_rgb = cv2.cvtColor(resized_zoom_out, cv2.COLOR_BGR2RGB)
    resized_specific_rgb = cv2.cvtColor(resized_specific, cv2.COLOR_BGR2RGB)

    # Plotting the images
    fig, axs = plt.subplots(1, 4, figsize=(20, 10))
    axs[0].imshow(image_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(resized_zoom_in_rgb)
    axs[1].set_title('Zoom In 2x')
    axs[1].axis('off')

    axs[2].imshow(resized_zoom_out_rgb)
    axs[2].set_title('Zoom Out 0.5x')
    axs[2].axis('off')

    axs[3].imshow(resized_specific_rgb)
    axs[3].set_title('Resize to 900x400')
    axs[3].axis('off')

    plt.show()
