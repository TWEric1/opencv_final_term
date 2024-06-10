from PyQt6 import QtWidgets, QtGui, QtCore
import cv2 as cv
import sys, webbrowser
import numpy as np

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('心碎舔狗的影像修改器')
        self.setWindowIcon(QtGui.QIcon('./icon.png'))
        self.startResolution()
        self.ui()
        self.original_image = None
        self.modified_image = None

    def startResolution(self):  # 初始化螢幕解析度
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        userWidth = screen_geometry.width()
        userHeight = screen_geometry.height()
        self.resize(userWidth // 2, userHeight // 2)  # 使用者螢幕解析度的1/2

    def ui(self):  # 顯示元素
        self.setStyleSheet('background:#333333;')
        layout = QtWidgets.QVBoxLayout(self)

        self.label_original = QtWidgets.QLabel("尚未匯入圖片", self)
        self.label_modified = QtWidgets.QLabel("尚未修改圖片", self)

        self.label_original.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_modified.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        img_layout = QtWidgets.QHBoxLayout()
        img_layout.addWidget(self.label_original)
        img_layout.addWidget(self.label_modified)
        layout.addLayout(img_layout)

        self.menubar = QtWidgets.QMenuBar(self)

        self.menu_file = HoverMenu('檔案')
        self.action_open = QtGui.QAction('開啟', self)
        self.action_open.triggered.connect(self.open)
        self.menu_file.addAction(self.action_open)
        self.action_save = QtGui.QAction('儲存', self)
        self.action_save.triggered.connect(self.save)
        self.menu_file.addAction(self.action_save)
        self.menu_file.addSeparator()
        self.action_close = QtGui.QAction('關閉', self)
        self.action_close.triggered.connect(self.close)
        self.menu_file.addAction(self.action_close)
        self.menubar.addMenu(self.menu_file)

        self.visual_effects = HoverMenu('影像處理')
        self.img_zoom = QtGui.QAction('影像縮放...', self)
        self.img_zoom.triggered.connect(self.open_zoom_window)
        self.visual_effects.addAction(self.img_zoom)
        self.spin = QtGui.QAction('旋轉...',self)
        self.spin.triggered.connect(self.open_rotate_window)
        self.visual_effects.addAction(self.spin)
        self.img_flip = HoverMenu('翻轉', self)
        self.img_level_filp = QtGui.QAction('左右翻轉',self)
        self.img_vertical_filp = QtGui.QAction('上下翻轉',self)
        self.img_vertical_and_level_filp = QtGui.QAction('左右上下翻轉',self)
        self.img_level_filp.triggered.connect(self.flip_level_image)
        self.img_vertical_filp.triggered.connect(self.flip_vertical_image)
        self.img_vertical_and_level_filp.triggered.connect(self.flip_vertical_and_level_image)
        self.img_flip.addActions([self.img_level_filp,self.img_vertical_filp,self.img_vertical_and_level_filp])
        self.visual_effects.addMenu(self.img_flip)
        self.grayscale = QtGui.QAction('灰階', self)
        self.grayscale.triggered.connect(self.convert_to_grayscale)
        self.visual_effects.addAction(self.grayscale)
        self.binary = HoverMenu('二值化', self)
        self.threshold = QtGui.QAction("固定閾值...",self)
        self.threshold.triggered.connect(self.open_threshold_window)
        self.adaptive_Gaussian_threshold = QtGui.QAction('自適應閾值(高斯法)',self)
        self.adaptive_Gaussian_threshold.triggered.connect(self.apply_adaptive_gaussian_threshold)
        self.adaptive_mean_threshold = QtGui.QAction('自適應閾值(平均法)',self)
        self.adaptive_mean_threshold.triggered.connect(self.apply_adaptive_mean_threshold)
        self.binary.addActions([self.threshold,self.adaptive_Gaussian_threshold,self.adaptive_mean_threshold])
        self.visual_effects.addMenu(self.binary)
        self.filter = HoverMenu('濾波',self)
        self.mean = QtGui.QAction('平均濾波', self)
        self.Gaussian = QtGui.QAction('高斯濾波', self)
        self.median = QtGui.QAction('中值濾波', self)
        self.bilateral = QtGui.QAction('雙邊濾波', self)
        self.mean.triggered.connect(self.apply_mean_filter)        
        self.Gaussian.triggered.connect(self.apply_gaussian_filter)   
        self.median.triggered.connect(self.apply_median_filter)    
        self.bilateral.triggered.connect(self.apply_bilateral_filter)
        self.filter.addActions([self.mean,self.Gaussian,self.median,self.bilateral])
        self.visual_effects.addMenu(self.filter)
        self.convolutional = QtGui.QAction('捲積...', self)
        self.convolutional.triggered.connect(self.open_convolution_window)
        self.visual_effects.addAction(self.convolutional)
        self.Histogram_Equalization = QtGui.QAction('直方圖均衡化', self)
        self.Histogram_Equalization.triggered.connect(self.apply_histogram_equalization)
        self.visual_effects.addAction(self.Histogram_Equalization)
        self.edge_detection = HoverMenu('邊緣偵測',self)
        self.Sobel_edge_detection = QtGui.QAction('Sobel邊緣偵測...', self)
        self.canny = QtGui.QAction('Canny邊緣偵測...',self)
        self.mixed_laplacian = QtGui.QAction('混合拉普拉斯邊緣偵測...',self)
        self.Sobel_edge_detection.triggered.connect(self.open_sobel_edge_detection_window)
        self.canny.triggered.connect(self.open_canny_edge_detection_window)
        self.mixed_laplacian.triggered.connect(self.open_mixed_laplacian_edge_detection_window)
        self.edge_detection.addActions([self.Sobel_edge_detection,self.canny,self.mixed_laplacian])
        self.visual_effects.addMenu(self.edge_detection)
        self.img_effects = HoverMenu('影像效果',self)
        self.contrast = QtGui.QAction('對比度...', self)
        self.contrast.triggered.connect(self.adjust_contrast)
        self.img_effects.addAction(self.contrast)
        self.img_color = HoverMenu('色調',self)
        self.warm_color = QtGui.QAction('暖色調',self)
        self.warm_color.triggered.connect(self.apply_warm_tone)
        self.cool_color =QtGui.QAction('冷色調',self)
        self.cool_color.triggered.connect(self.apply_cool_tone)
        self.img_color.addActions([self.warm_color,self.cool_color])
        self.img_effects.addMenu(self.img_color)
        self.sharp_effects = HoverMenu('影像銳化',self)
        self.SUM_sharp = QtGui.QAction('USM銳利化',self)
        self.SUM_sharp.triggered.connect(self.apply_usm_sharpen)
        self.Laplacian_sharp = QtGui.QAction('拉普拉斯銳利化',self)
        self.Laplacian_sharp.triggered.connect(self.apply_laplacian_sharpen)
        self.sharp_effects.addActions([self.SUM_sharp,self.Laplacian_sharp])
        self.img_effects.addMenu(self.sharp_effects)
        self.negative = QtGui.QAction('影像負片',self)
        self.negative.triggered.connect(self.apply_negative_image)
        self.noise = HoverMenu('雜訊',self)
        self.Gaussian_noise = QtGui.QAction('高斯雜訊...',self)
        self.Gaussian_noise.triggered.connect(self.add_gaussian_noise)
        self.salt_and_pepper_noise = QtGui.QAction('椒鹽雜訊...',self)
        self.salt_and_pepper_noise.triggered.connect(self.add_salt_and_pepper_noise)
        self.denoise = QtGui.QAction('去雜訊...',self)
        self.denoise.triggered.connect(self.open_denoise_window)
        self.noise.addActions([self.Gaussian_noise,self.salt_and_pepper_noise,self.denoise])
        self.img_effects.addMenu(self.noise)
        self.erode = QtGui.QAction('侵蝕...',self)
        self.erode.triggered.connect(self.open_erode_window)
        self.dilate = QtGui.QAction('膨脹...',self)
        self.dilate.triggered.connect(self.open_dilate_window)
        self.Disconnect = QtGui.QAction('斷開...',self)
        self.Disconnect.triggered.connect(self.open_Disconnect_window)
        self.closing = QtGui.QAction('閉合...',self)
        self.closing.triggered.connect(self.open_closing_window)

        self.img_effects.addActions([self.negative,self.erode,self.dilate,self.Disconnect,self.closing])
        self.visual_effects.addMenu(self.img_effects)
        self.menubar.addMenu(self.visual_effects)

        self.instructions = HoverMenu('說明')
        self.about = QtGui.QAction('關於我', self)
        self.about.triggered.connect(self.openWeb)
        self.instructions.addAction(self.about)
        '''
        self.version = QtGui.QAction('版本',self)
        self.instructions.addAction(self.version)
        '''
        self.menubar.addMenu(self.instructions)

        self.menubar.setStyleSheet('background:#FFFFFF;height: 35px;font-size: 10px;color: black;')
        self.menu_file.setStyleSheet("""
            QMenu {
                background-color: #FFFFFF;
                                     color: black;
            }
            QMenu::item {
                background-color: #FFFFFF;
                color: black;
            }
        """)

        layout.setMenuBar(self.menubar)

        button_layout = QtWidgets.QHBoxLayout() 
        layout.addLayout(button_layout)

        self.btnOpen = HoverButton('開啟圖片', self)
        self.btnOpen.clicked.connect(self.open)
        button_layout.addWidget(self.btnOpen)

        self.btnSave = HoverButton('儲存圖片', self)
        self.btnSave.clicked.connect(self.save)
        button_layout.addWidget(self.btnSave)

        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.confirm)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.cannal)
        button_layout.addWidget(self.btnCannal)

    def open(self):  # 選擇圖片
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, '開啟圖片', '', 'Images (*.png *.jpg *.bmp)')
        if filePath:
            self.original_image = cv.imread(filePath)
            if self.original_image is not None:
                self.show_image(self.original_image, self.label_original)
                self.label_modified.setText("尚未修改圖片")
                self.modified_image = None
            else:
                print("無法加載圖片，請檢查文件路徑。")
                self.label_original.setText("無法加載圖片")

    def save(self):  # 儲存圖片
        if self.original_image is not None:
            filePath, _ = QtWidgets.QFileDialog.getSaveFileName(self, '儲存圖片', '', 'Images (*.png *.jpg *.bmp)')
            if filePath:
                cv.imwrite(filePath, self.original_image)
                print("圖片已儲存至:", filePath)
        else:
            print("沒有可儲存的圖片。")

    def show_image(self, img, label):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(img_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def close(self):  # 關閉程式
        QtWidgets.QApplication.quit()

    def convert_to_grayscale(self):  # 灰階化圖片
        if self.original_image is not None:
            self.modified_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            self.modified_image = cv.cvtColor(self.modified_image, cv.COLOR_GRAY2BGR)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_zoom_window(self):
        if self.original_image is not None:
            dialog = ZoomWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                scale_factor = dialog.slider.value() / 100.0
                self.modified_image = cv.resize(self.original_image, None, fx=scale_factor, fy=scale_factor)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    # 影像旋轉功能
    def open_rotate_window(self):
        if self.original_image is not None:
            dialog = RotateWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                angle = dialog.slider.value()
                (h, w) = self.original_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv.getRotationMatrix2D(center, angle, 1.0)
                self.modified_image = cv.warpAffine(self.original_image, M, (w, h))
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def flip_level_image(self):  # 翻轉左右圖片
        if self.original_image is not None:
            self.modified_image = cv.flip(self.original_image, 1)  # 1表示水平翻轉，0表示垂直翻轉 -1表示上下水平翻轉
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")
    
    def flip_vertical_image(self):  # 翻轉上下圖片
        if self.original_image is not None:
            self.modified_image = cv.flip(self.original_image, 0)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def flip_vertical_and_level_image(self):  # 翻轉上下水平圖片
        if self.original_image is not None:
            self.modified_image = cv.flip(self.original_image, -1)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_threshold_window(self):
        if self.original_image is not None:
            dialog = ThresholdWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                threshold_value = dialog.slider.value()
                gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
                _, self.modified_image = cv.threshold(gray_image, threshold_value, 255, cv.THRESH_BINARY)
                self.modified_image = cv.cvtColor(self.modified_image, cv.COLOR_GRAY2BGR)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_adaptive_gaussian_threshold(self):
        if self.original_image is not None:
            gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            self.modified_image = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            self.modified_image = cv.cvtColor(self.modified_image, cv.COLOR_GRAY2BGR)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_adaptive_mean_threshold(self):
        if self.original_image is not None:
            gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            self.modified_image = cv.adaptiveThreshold(
            gray_image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
            self.modified_image = cv.cvtColor(self.modified_image, cv.COLOR_GRAY2BGR)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_mean_filter(self):
        if self.original_image is not None:
            self.modified_image = cv.blur(self.original_image, (3, 3))
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_gaussian_filter(self):
        if self.original_image is not None:
            self.modified_image = cv.GaussianBlur(self.original_image, (3, 3), 0)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_median_filter(self):
        if self.original_image is not None:
            self.modified_image = cv.medianBlur(self.original_image, 3)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_bilateral_filter(self):
        if self.original_image is not None:
            self.modified_image = cv.bilateralFilter(self.original_image, 9, 75, 75)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_histogram_equalization(self):
        if self.original_image is not None:
            gray_image = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            equalized_image = cv.equalizeHist(gray_image)
            self.modified_image = cv.cvtColor(equalized_image, cv.COLOR_GRAY2BGR)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")
        
    def open_convolution_window(self):
        if self.original_image is not None:
            dialog = ConvolutionWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                kernel_size = dialog.get_kernel_size()
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
                self.modified_image = cv.filter2D(self.original_image, -1, kernel)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_sobel_edge_detection_window(self):
        if self.original_image is not None:
            dialog = SobelEdgeDetectionWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                ksize = dialog.get_ksize()
                sobelx = cv.Sobel(self.original_image, cv.CV_64F, 1, 0, ksize=ksize)
                sobely = cv.Sobel(self.original_image, cv.CV_64F, 0, 1, ksize=ksize)
                self.modified_image = cv.magnitude(sobelx, sobely)
                self.modified_image = cv.convertScaleAbs(self.modified_image)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_canny_edge_detection_window(self):
        if self.original_image is not None:
            dialog = CannyEdgeDetectionWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                threshold1, threshold2 = dialog.get_thresholds()
                self.modified_image = cv.Canny(self.original_image, threshold1, threshold2)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_mixed_laplacian_edge_detection_window(self):
        if self.original_image is not None:
            dialog = MixedLaplacianEdgeDetectionWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                ksize = dialog.get_ksize()
                laplacian = cv.Laplacian(self.original_image, cv.CV_64F, ksize=ksize)
                self.modified_image = cv.convertScaleAbs(laplacian)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def adjust_contrast(self):
        if self.original_image is not None:
            dialog = ContrastWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                contrast = dialog.slider.value() / 100.0
                self.modified_image = cv.convertScaleAbs(self.original_image, alpha=contrast, beta=0)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_warm_tone(self):
        if self.original_image is not None:
            increase = 50
            self.modified_image = cv.addWeighted(self.original_image, 1, np.zeros(self.original_image.shape, self.original_image.dtype), 0, increase)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_cool_tone(self):
        if self.original_image is not None:
            increase = 50
            self.modified_image = cv.addWeighted(self.original_image, 1, np.zeros(self.original_image.shape, self.original_image.dtype), 0, -increase)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_usm_sharpen(self):
        if self.original_image is not None:
            img_float = self.original_image.astype(np.float32) / 255.0
            blurred = cv.GaussianBlur(img_float, (9, 9), 10.0)
            sharpened = cv.addWeighted(img_float, 1.5, blurred, -0.5, 0)
            self.modified_image = np.clip(sharpened * 255.0, 0, 255).astype(np.uint8)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_laplacian_sharpen(self):
        if self.original_image is not None:
            gray = cv.cvtColor(self.original_image, cv.COLOR_BGR2GRAY)
            laplacian = cv.Laplacian(gray, cv.CV_64F)
            laplacian = cv.convertScaleAbs(laplacian)
            self.modified_image = cv.add(self.original_image, cv.cvtColor(laplacian, cv.COLOR_GRAY2BGR))
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def add_gaussian_noise(self):
        if self.original_image is not None:
            dialog = GaussianNoiseWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                noise_level = dialog.slider.value()
                row, col, ch = self.original_image.shape
                mean = 0
                sigma = noise_level ** 0.5
                gauss = np.random.normal(mean, sigma, (row, col, ch))
                gauss = gauss.reshape(row, col, ch)
                self.modified_image = self.original_image + gauss
                self.modified_image = np.clip(self.modified_image, 0, 255).astype(np.uint8)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def add_salt_and_pepper_noise(self):
        if self.original_image is not None:
            dialog = SaltPepperNoiseWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                noise_level = dialog.slider.value() / 100.0
                row, col, ch = self.original_image.shape
                s_vs_p = 0.5
                amount = noise_level
                out = np.copy(self.original_image)
                num_salt = np.ceil(amount * self.original_image.size * s_vs_p)
                num_pepper = np.ceil(amount * self.original_image.size * (1.0 - s_vs_p))

                # Salt mode
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.original_image.shape]
                out[coords[0], coords[1], :] = 1

                # Pepper mode
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.original_image.shape]
                out[coords[0], coords[1], :] = 0

                self.modified_image = out
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_denoise_window(self):
        if self.original_image is not None:
            dialog = DenoiseWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                h, w = self.original_image.shape[:2]
                denoise_value = dialog.get_denoise_value()
                self.modified_image = cv.fastNlMeansDenoisingColored(self.original_image, None, h * denoise_value / 100.0, w * denoise_value / 100.0, 7, 21)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def apply_negative_image(self):
        if self.original_image is not None:
            self.modified_image = cv.bitwise_not(self.original_image)
            self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_erode_window(self):
        if self.original_image is not None:
            dialog = ErodeWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                kernel_size = dialog.get_erode_value()
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.modified_image = cv.erode(self.original_image, kernel, iterations=1)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_dilate_window(self):
        if self.original_image is not None:
            dialog = DilateWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                kernel_size = dialog.get_dilate_value()
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.modified_image = cv.dilate(self.original_image, kernel, iterations=1)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_Disconnect_window(self):
        if self.original_image is not None:
            dialog = DisconnectWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                kernel_size = dialog.get_open_value()
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.modified_image = cv.morphologyEx(self.original_image, cv.MORPH_OPEN, kernel)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")

    def open_closing_window(self):
        if self.original_image is not None:
            dialog = closingWindow(self)
            if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                kernel_size = dialog.get_close_value()
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.modified_image = cv.morphologyEx(self.original_image, cv.MORPH_CLOSE, kernel)
                self.show_image(self.modified_image, self.label_modified)
        else:
            print("尚未匯入圖片")
                
    def confirm(self):  #確定按鈕功能
        if self.modified_image is not None:
            self.original_image = self.modified_image.copy()
            self.show_image(self.original_image, self.label_original)
            self.label_modified.setText("尚未修改圖片")
            self.modified_image = None
        else:
            print("沒有可確定的修改圖片")

    def cannal(self): #取消按鈕功能
        if self.modified_image is not None:
            self.label_modified.setText("尚未修改圖片")
            self.modified_image = None
        else:
            print("沒有可取消的修改圖片")

    def openWeb(self):
        webbrowser.get('windows-default').open_new('https://github.com/TWEric1')

class ZoomWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('影像縮放')
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(10, 300)  # 100表示100%
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_label)
        
        self.label = QtWidgets.QLabel('100%', self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet('color: white;')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)
        
        self.layout.addLayout(self.button_layout)
    
    def update_label(self, value):
        self.label.setText(f'{value}%')

class RotateWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('旋轉影像')
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 360)  # 0到360度
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_label)
        
        self.label = QtWidgets.QLabel('0°', self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet('color: white;')
        
        self.slider_layout = QtWidgets.QHBoxLayout()
        self.slider_layout.addStretch()
        self.slider_layout.addWidget(self.label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addStretch()
        
        self.layout.addLayout(self.slider_layout)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)
        
        self.layout.addLayout(self.button_layout)
    
    def update_label(self, value):
        self.label.setText(f'{value}°')
    
class ThresholdWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('固定閥值')
        self.layout = QtWidgets.QVBoxLayout(self)
        
        self.max_value_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.max_value_slider.setRange(1, 255)
        self.max_value_slider.setValue(255)
        self.max_value_slider.valueChanged.connect(self.update_max_label)
        
        self.max_value_label = QtWidgets.QLabel('255', self)
        self.max_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.max_value_label.setStyleSheet('color: white;')
        self.layout.addWidget(self.max_value_label)
        self.layout.addWidget(self.max_value_slider)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setRange(0, 255)
        self.slider.setValue(127)
        self.slider.valueChanged.connect(self.update_label)
        
        self.label = QtWidgets.QLabel('127', self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet('color: white;')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)
        
        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)
        
        self.layout.addLayout(self.button_layout)
    
    def update_label(self, value):
        self.label.setText(f'{value}')
    
    def update_max_label(self, value):
        self.max_value_label.setText(f'{value}')
        self.slider.setRange(0, value)
    
class ConvolutionWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('捲積')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.kernel_size_label = QtWidgets.QLabel('Kernel Size:', self)
        self.kernel_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.kernel_size_slider.setRange(1, 31)
        self.kernel_size_slider.setTickInterval(2)
        self.kernel_size_slider.setValue(3)
        self.kernel_size_slider.setSingleStep(2)
        self.kernel_size_slider.valueChanged.connect(self.update_kernel_size_label)

        self.kernel_size_value_label = QtWidgets.QLabel('3', self)

        self.layout.addWidget(self.kernel_size_label)
        self.layout.addWidget(self.kernel_size_value_label)
        self.layout.addWidget(self.kernel_size_slider)
        
        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_kernel_size_label(self):
        ksize = self.kernel_size_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        self.kernel_size_value_label.setText(str(ksize))

    def get_kernel_size(self):
        ksize = self.kernel_size_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        return ksize

class SobelEdgeDetectionWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sobel 邊緣偵測')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.ksize_label = QtWidgets.QLabel('Kernel Size:', self)
        self.ksize_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.ksize_slider.setRange(1, 31)  # 通常大小範圍
        self.ksize_slider.setTickInterval(2)
        self.ksize_slider.setValue(3)
        self.ksize_slider.setSingleStep(2)
        self.ksize_slider.valueChanged.connect(self.update_ksize_label)

        self.ksize_value_label = QtWidgets.QLabel('3', self)

        self.layout.addWidget(self.ksize_label)
        self.layout.addWidget(self.ksize_slider)
        self.layout.addWidget(self.ksize_value_label)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_ksize_label(self):
        ksize = self.ksize_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        self.ksize_value_label.setText(str(ksize))

    def get_ksize(self):
        ksize = self.ksize_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        return ksize
    
class CannyEdgeDetectionWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Canny 邊緣偵測')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.threshold1_label = QtWidgets.QLabel('Threshold 1:', self)
        self.threshold1_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.threshold1_slider.setRange(0, 255)
        self.threshold1_slider.setValue(50)
        self.threshold1_slider.valueChanged.connect(self.update_threshold1_label)

        self.threshold1_value_label = QtWidgets.QLabel('50', self)

        self.threshold2_label = QtWidgets.QLabel('Threshold 2:', self)
        self.threshold2_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.threshold2_slider.setRange(0, 255)
        self.threshold2_slider.setValue(150)
        self.threshold2_slider.valueChanged.connect(self.update_threshold2_label)

        self.threshold2_value_label = QtWidgets.QLabel('150', self)

        self.layout.addWidget(self.threshold1_label)
        self.layout.addWidget(self.threshold1_slider)
        self.layout.addWidget(self.threshold1_value_label)
        self.layout.addWidget(self.threshold2_label)
        self.layout.addWidget(self.threshold2_slider)
        self.layout.addWidget(self.threshold2_value_label)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

        self.threshold2_slider.valueChanged.connect(self.update_threshold1_max_value)
        self.threshold1_slider.valueChanged.connect(self.update_threshold2_min_value)

    def update_threshold1_label(self):
        self.threshold1_value_label.setText(str(self.threshold1_slider.value()))

    def update_threshold2_label(self):
        self.threshold2_value_label.setText(str(self.threshold2_slider.value()))

    def update_threshold1_max_value(self):
        self.threshold1_slider.setMaximum(self.threshold2_slider.value())
        if self.threshold1_slider.value() > self.threshold2_slider.value():
            self.threshold1_slider.setValue(self.threshold2_slider.value())

    def update_threshold2_min_value(self):
        self.threshold2_slider.setMinimum(self.threshold1_slider.value())
        if self.threshold2_slider.value() < self.threshold1_slider.value():
            self.threshold2_slider.setValue(self.threshold1_slider.value())

    def get_thresholds(self):
        return self.threshold1_slider.value(), self.threshold2_slider.value()
    
class MixedLaplacianEdgeDetectionWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('混合拉普拉斯邊緣偵測')
        self.layout = QtWidgets.QVBoxLayout(self)

        self.ksize_label = QtWidgets.QLabel('Kernel Size:', self)
        self.ksize_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.ksize_slider.setRange(1, 31)
        self.ksize_slider.setTickInterval(2)
        self.ksize_slider.setValue(3)
        self.ksize_slider.setSingleStep(2)
        self.ksize_slider.valueChanged.connect(self.update_ksize_label)

        self.ksize_value_label = QtWidgets.QLabel('3', self)

        self.layout.addWidget(self.ksize_label)
        self.layout.addWidget(self.ksize_slider)
        self.layout.addWidget(self.ksize_value_label)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_ksize_label(self):
        ksize = self.ksize_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        self.ksize_value_label.setText(str(ksize))

    def get_ksize(self):
        ksize = self.ksize_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        return ksize
    
class ContrastWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('對比度')
        self.setGeometry(200, 200, 300, 100)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(200)
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.update_label)

        self.label = QtWidgets.QLabel('對比度: 1.0')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_label(self):
        contrast_value = self.slider.value() / 100.0
        self.label.setText(f'對比度: {contrast_value:.2f}')


class GaussianNoiseWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('添加高斯雜訊')
        self.setGeometry(200, 200, 300, 100)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.update_label)

        self.label = QtWidgets.QLabel('雜訊強度: 10')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_label(self):
        self.label.setText(f'雜訊強度: {self.slider.value()}')


class SaltPepperNoiseWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('添加椒鹽雜訊')
        self.setGeometry(200, 200, 300, 100)
        self.layout = QtWidgets.QVBoxLayout(self)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal,self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(5)
        self.slider.valueChanged.connect(self.update_label)

        self.label = QtWidgets.QLabel('雜訊強度: 5%')
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.btnConfirm = StyledButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        self.btnCancel = StyledButton('取消', self)
        self.btnCancel.clicked.connect(self.reject)

        self.button_layout.addWidget(self.btnConfirm)
        self.button_layout.addWidget(self.btnCancel)

        self.layout.addLayout(self.button_layout)

    def update_label(self):
        self.label.setText(f'雜訊強度: {self.slider.value()}%')

class DenoiseWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('去雜訊')
        self.setFixedSize(300, 100)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(3)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f'去雜訊程度: {self.slider.value()}')
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.reject)
        button_layout.addWidget(self.btnCannal)
        
        layout.addLayout(button_layout)

    def update_label(self, value):
        self.label.setText(f'去雜訊程度: {value}')

    def get_denoise_value(self):
        return self.slider.value()
    
class ErodeWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('侵蝕')
        self.setFixedSize(300, 100)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 20)
        self.slider.setValue(3)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f'侵蝕核大小: {self.slider.value()}')
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.reject)
        button_layout.addWidget(self.btnCannal)
        
        layout.addLayout(button_layout)

    def update_label(self, value):
        self.label.setText(f'侵蝕核大小: {value}')

    def get_erode_value(self):
        return self.slider.value()

class DilateWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('膨脹')
        self.setFixedSize(300, 100)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 20)
        self.slider.setValue(3)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f'膨脹核大小: {self.slider.value()}')
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.reject)
        button_layout.addWidget(self.btnCannal)
        
        layout.addLayout(button_layout)

    def update_label(self, value):
        self.label.setText(f'膨脹核大小: {value}')

    def get_dilate_value(self):
        return self.slider.value()
    
class DisconnectWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('斷開')
        self.setFixedSize(300, 100)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 20)
        self.slider.setValue(3)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f'斷開核大小: {self.slider.value()}')
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.reject)
        button_layout.addWidget(self.btnCannal)
        
        layout.addLayout(button_layout)

    def update_label(self, value):
        self.label.setText(f'斷開核大小: {value}')

    def get_open_value(self):
        return self.slider.value()


class closingWindow(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('閉合')
        self.setFixedSize(300, 100)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(1, 20)
        self.slider.setValue(3)
        self.slider.setTickInterval(1)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        layout.addWidget(self.slider)
        
        self.label = QtWidgets.QLabel(f'閉合核大小: {self.slider.value()}')
        layout.addWidget(self.label)
        
        self.slider.valueChanged.connect(self.update_label)
        
        button_layout = QtWidgets.QHBoxLayout()
        
        self.btnConfirm = HoverButton('確定', self)
        self.btnConfirm.clicked.connect(self.accept)
        button_layout.addWidget(self.btnConfirm)

        self.btnCannal = HoverButton('取消', self)
        self.btnCannal.clicked.connect(self.reject)
        button_layout.addWidget(self.btnCannal)
        
        layout.addLayout(button_layout)

    def update_label(self, value):
        self.label.setText(f'閉合核大小: {value}')

    def get_close_value(self):
        return self.slider.value()

class HoverButton(QtWidgets.QPushButton):  #滑鼠指到按鈕的效果顯示
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet('background-color: white; color: black;')
        self.normal_color = 'background-color: white; color: black;'
        self.hover_color = 'background-color: lightblue; color: black;'
        self.setMouseTracking(True)

    def enterEvent(self, event):
        self.setStyleSheet(self.hover_color)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.setStyleSheet(self.normal_color)
        super().leaveEvent(event)

class StyledButton(HoverButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)

class HoverMenu(QtWidgets.QMenu):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet('background-color: white; color: black;')
        self.normal_color = 'background-color: white; color: black;'
        self.hover_color = 'background-color: lightblue;'

    def enterEvent(self, event):
        self.setStyleSheet(self.hover_color)

    def leaveEvent(self, event):
        self.setStyleSheet(self.normal_color)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec())