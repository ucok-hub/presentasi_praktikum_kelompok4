import cv2


def crop_image(image_path, start_row, start_col, end_row, end_col):
    # Baca citra dari file
    image = cv2.imread(image_path)

    if image is None:
        print("Gambar tidak ditemukan atau path tidak valid.")
        return None

    # Potong citra sesuai koordinat yang ditentukan
    cropped_image = image[start_row:end_row, start_col:end_col]

    # Tampilkan citra hasil crop
    cv2.imshow('Cropped Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_image


# Path gambar dengan tanda backslash ganda
image_path = r'C:\\Users\\user\\OneDrive - Office 365 Original\\Documents\\----- PENTING -----\\TUGAS-TUGAS\\PNJ SEMESTER 4\\PENGOLAHAN CITRA DIGITAL\\PRAKTIKUM\\CHAPTER B\\b04.cropimage\\peter.jpg'
start_row, start_col = 50, 100
end_row, end_col = 200, 300

cropped_image = crop_image(image_path, start_row, start_col, end_row, end_col)
