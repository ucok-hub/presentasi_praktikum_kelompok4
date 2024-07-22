import numpy as np
import sys
import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt  # Import library matplotlib


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Stretching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner_Image.triggered.connect(self.binerImage)
        self.actionHistogram_Gray.triggered.connect(self.histogramGrayScale)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogramClicked)
        self.actionHistogram_Equilization.triggered.connect(self.EqualHistogramClicked)
        self.actionTranslation.triggered.connect(self.translasi)

        # Hubungkan action Rotasi ke metode rotasi
        self.actionMin45_deg.triggered.connect(lambda: self.rotasi(-45))
        self.action45_deg.triggered.connect(lambda: self.rotasi(45))
        self.actionMin90_deg.triggered.connect(lambda: self.rotasi(-90))
        self.action90_deg.triggered.connect(lambda: self.rotasi(90))
        self.action180_deg.triggered.connect(lambda: self.rotasi(180))

        # Hubungkan action Resize ke metode resize
        self.actionzoom_in.triggered.connect(self.zoomIn)
        self.actionzoom_out.triggered.connect(self.zoomOut)
        self.actionskew.triggered.connect(self.skewedImage)

        # Hubungkan action Crop ke metode cropImage
        self.actionCrop.triggered.connect(self.cropImage)

        # Hubungkan action Add dan Subtract ke metode addImages dan subtractImages
        self.actionAdd.triggered.connect(self.addImages)
        self.actionSubtract.triggered.connect(self.subtractImages)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('koala.jpeg')

    def loadImage(self, flname):
        try:
            self.image = cv2.imread(flname)
            if self.image is None:
                raise FileNotFoundError(f"Could not load image: {flname}")
            self.displayImage()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def grayClicked(self):
        try:
            if self.image is not None:
                H, W = self.image.shape[:2]
                gray = np.zeros((H, W), np.uint8)
                for i in range(H):
                    for j in range(W):
                        gray[i, j] = np.clip(
                            0.299 * self.image[i, j, 0] + 0.587 * self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0,
                            255)
                self.image = gray
                self.displayImage(2)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def brightness(self):
        try:
            if self.image is not None:
                brightness = 50
                self.image = np.clip(self.image.astype(int) + brightness, 0, 255).astype(np.uint8)
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrast(self):
        try:
            if self.image is not None:
                contrast = 1.6
                img_float = self.image.astype(float)
                img_contrast = np.clip(img_float * contrast, 0, 255).astype(np.uint8)
                self.image = img_contrast
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def contrastStretching(self):
        try:
            if self.image is not None:
                max_pixel = np.max(self.image)
                min_pixel = np.min(self.image)
                dynamic_range = max_pixel - min_pixel
                stretched_image = ((self.image - min_pixel) / dynamic_range) * 255
                stretched_image = np.clip(stretched_image, 0, 255).astype(np.uint8)
                self.image = stretched_image
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def negativeImage(self):
        try:
            if self.image is not None:
                negative_image = 255 - self.image
                self.image = negative_image
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def binerImage(self):
        try:
            if self.image is not None:
                gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, biner_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                self.image = biner_image
                self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def histogramGrayScale(self):
        try:
            if self.image is not None:
                # Konversi citra ke grayscale jika tidak sudah dalam grayscale
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image
                self.image = gray
                self.displayImage(1)

                # Tampilkan histogram menggunakan matplotlib
                plt.hist(gray.ravel(), 255, [0, 255])
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.title('Histogram of Grayscale Image')
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def RGBHistogramClicked(self):
        try:
            if self.image is not None:
                color = ('b', 'g', 'r')
                for i, col in enumerate(color):
                    histo = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                    plt.plot(histo, color=col)
                    plt.xlim([0, 256])
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.title('Histogram of RGB Image')
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def EqualHistogramClicked(self):
        try:
            if self.image is not None:
                if len(self.image.shape) == 3:
                    gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.image

                hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = cdf * hist.max() / cdf.max()
                cdf_m = np.ma.masked_equal(cdf, 0)
                cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
                cdf = np.ma.filled(cdf_m, 0).astype('uint8')
                self.image = cdf[gray]
                self.displayImage(2)

                plt.plot(cdf_normalized, color='b')
                plt.hist(self.image.flatten(), 256, [0, 256], color='r')
                plt.xlim([0, 256])
                plt.xlabel('Pixel Value')
                plt.ylabel('Frequency')
                plt.title('Histogram Equalization')
                plt.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def translasi(self):
        try:
            if self.image is not None:
                h, w = self.image.shape[:2]
                quarter_h, quarter_w = h / 4, w / 4
                T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
                translated_image = cv2.warpAffine(self.image, T, (w, h))
                self.image = translated_image
                self.displayImage(2)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def rotasi(self, degree):
        try:
            # Memeriksa apakah gambar sudah ada
            if self.image is not None:
                # Mengambil tinggi dan lebar gambar
                h, w = self.image.shape[:2]
                # Menentukan titik pusat gambar
                center = (w // 2, h // 2)

                # Membuat matriks rotasi berdasarkan pusat dan derajat yang diinginkan
                rotationMatrix = cv2.getRotationMatrix2D(center, degree, scale=1.0)

                # Menghitung nilai absolut dari kosinus dan sinus sudut rotasi
                abs_cos = abs(rotationMatrix[0, 0])
                abs_sin = abs(rotationMatrix[0, 1])

                # Menghitung batas baru dari lebar dan tinggi gambar setelah rotasi
                bound_w = int(h * abs_sin + w * abs_cos)
                bound_h = int(h * abs_cos + w * abs_sin)

                # Menyesuaikan matriks rotasi dengan menambahkan pergeseran untuk menyesuaikan batas baru
                rotationMatrix[0, 2] += bound_w // 2 - center[0]
                rotationMatrix[1, 2] += bound_h // 2 - center[1]

                # Menerapkan rotasi pada gambar dengan menggunakan warpAffine
                rotated_image = cv2.warpAffine(self.image, rotationMatrix, (bound_w, bound_h))
                # Menyimpan gambar yang sudah dirotasi kembali ke atribut self.image
                self.image = rotated_image
                # Menampilkan gambar yang sudah dirotasi
                self.displayImage(2)

        # Menangkap dan menampilkan pesan kesalahan jika terjadi exception
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomIn(self):
        try:
            if self.image is not None:
                resize_img = cv2.resize(self.image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Zoom In', resize_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def zoomOut(self):
        try:
            if self.image is not None:
                resize_img = cv2.resize(self.image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow('Zoom Out', resize_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def skewedImage(self):
        try:
            if self.image is not None:
                resize_img = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA)
                cv2.imshow('Skewed Image', resize_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def cropImage(self):
        try:
            if self.image is not None:
                # Tentukan koordinat atau posisi x (row) dan y (coloum) awal yang diawali dari ujung kiri atas
                start_row = 50
                start_col = 50

                # Tentukan koordinat atau posisi x (row) dan y (coloum) akhir berakhir di ujung kanan bawah
                end_row = 300
                end_col = 500

                # Set koordinat image citra -> citra [start row s/d end row, start col s/d end col]
                cropped_image = self.image[start_row:end_row, start_col:end_col]

                # Tampilkan citra
                cv2.imshow('Cropped Image', cropped_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def addImages(self):
        try:
            img1 = cv2.imread('koala.jpeg', 0)
            img2 = cv2.imread('harimau.jpeg', 0)
            if img1 is None or img2 is None:
                raise FileNotFoundError("Could not load one or both images: img1.jpg, img2.jpg")

            # Resize img2 to match the size of img1
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            add_img = cv2.add(img1, img2_resized)
            cv2.imshow('Added Image', add_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def subtractImages(self):
        try:
            img1 = cv2.imread('koala.jpeg', 0)
            img2 = cv2.imread('harimau.jpeg', 0)
            if img1 is None or img2 is None:
                raise FileNotFoundError("Could not load one or both images: img1.jpg, img2.jpg")

            # Resize img2 to match the size of img1
            img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            subtract_img = cv2.subtract(img1, img2_resized)
            cv2.imshow('Subtracted Image', subtract_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, window=1):
        if self.image is not None:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
            img = img.rgbSwapped()
            pixmap = QPixmap.fromImage(img)

            if window == 1:
                self.imgLabel.setPixmap(pixmap)
                self.imgLabel.setAlignment(QtCore.Qt.AlignCenter)
                self.imgLabel.setScaledContents(True)
                self.imgLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                self.imgLabel.setMinimumSize(1, 1)
            elif window == 2:
                self.hasilLabel.setPixmap(pixmap)
                self.hasilLabel.setAlignment(QtCore.Qt.AlignCenter)
                self.hasilLabel.setScaledContents(True)
                self.hasilLabel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
                self.hasilLabel.setMinimumSize(1, 1)
        else:
            QMessageBox.critical(self, "Error", "No image loaded.")


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())
