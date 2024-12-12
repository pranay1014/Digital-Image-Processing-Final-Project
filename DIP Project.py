#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python numpy pillow


# In[14]:


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scale, Label, Button, HORIZONTAL
from PIL import Image, ImageTk, ImageEnhance

class ImageEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Adjustment Tool with OpenCV")

        self.image = None
        self.cv_image = None  # OpenCV image (numpy array)
        self.tk_image = None
        self.file_path = None

        # Main Layout
        self.canvas = tk.Label(self.root)
        self.canvas.pack()

        # Controls
        self.create_controls()

    def create_controls(self):
        # Buttons for Loading and Saving Images
        load_button = Button(self.root, text="Open Image", command=self.load_image)
        load_button.pack(side="top")

        save_button = Button(self.root, text="Save Image", command=self.save_image)
        save_button.pack(side="top")

        # Adjustment Sliders
        self.add_slider("Brightness", self.adjust_brightness, 0.5, 2, 1.0)
        self.add_slider("Contrast", self.adjust_contrast, 0.5, 2, 1.0)
        self.add_slider("Saturation", self.adjust_saturation, 0.5, 2, 1.0)
        self.add_slider("Hue", self.adjust_hue, -180, 180, 0)
        self.add_slider("Gamma", self.adjust_gamma, 0.1, 3, 1.0)

        # Filters
        Button(self.root, text="Black & White", command=self.apply_black_white).pack()
        Button(self.root, text="Sepia", command=self.apply_sepia).pack()
        Button(self.root, text="Vignette", command=self.apply_vignette).pack()
        Button(self.root, text="Blur", command=self.apply_blur).pack()
        Button(self.root, text="Sharpen", command=self.apply_sharpen).pack()

        # Cropping, Resizing, Rotations
        Button(self.root, text="Crop Image", command=self.crop_image).pack()
        Button(self.root, text="Resize Image", command=self.resize_image).pack()
        Button(self.root, text="Rotate 90°", command=lambda: self.rotate_image(90)).pack()
        Button(self.root, text="Rotate 180°", command=lambda: self.rotate_image(180)).pack()
        Button(self.root, text="Flip Horizontal", command=self.flip_horizontal).pack()
        Button(self.root, text="Flip Vertical", command=self.flip_vertical).pack()

    def add_slider(self, label, command, from_, to, init):
        frame = tk.Frame(self.root)
        frame.pack()
        label = Label(frame, text=label)
        label.pack(side="left")
        slider = Scale(frame, from_=from_, to=to, resolution=0.1, orient=HORIZONTAL, command=command)
        slider.set(init)
        slider.pack(side="right")

    def load_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.cv_image = cv2.imread(self.file_path)  # Load image as a numpy array
            self.image = Image.fromarray(cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB))  # Convert to PIL
            self.display_image()

    def save_image(self):
        if self.cv_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, self.cv_image)

    def display_image(self):
        if self.cv_image is not None:
            resized = cv2.resize(self.cv_image, (400, 400))  # Resize for display
            display_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)  # Convert to RGB for Tkinter
            self.tk_image = ImageTk.PhotoImage(Image.fromarray(display_image))
            self.canvas.config(image=self.tk_image)
            self.canvas.image = self.tk_image

    # Adjustments
    def adjust_brightness(self, value):
        if self.cv_image is not None:
            alpha = float(value)
            self.cv_image = cv2.convertScaleAbs(self.cv_image, alpha=alpha, beta=0)
            self.display_image()

    def adjust_contrast(self, value):
        if self.cv_image is not None:
            alpha = float(value)
            self.cv_image = cv2.convertScaleAbs(self.cv_image, alpha=alpha)
            self.display_image()

    def adjust_saturation(self, value):
        if self.cv_image is not None:
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], int(float(value) * 50))  # Adjust saturation
            self.cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.display_image()

    def adjust_hue(self, value):
        if self.cv_image is not None:
            hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] = cv2.add(hsv[:, :, 0], int(float(value)))  # Adjust hue
            self.cv_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            self.display_image()

    def adjust_gamma(self, value):
        if self.cv_image is not None:
            gamma = float(value)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            self.cv_image = cv2.LUT(self.cv_image, table)
            self.display_image()

    # Filters
    def apply_black_white(self):
        if self.cv_image is not None:
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2BGR)  # Keep 3 channels
            self.display_image()

    def apply_sepia(self):
        if self.cv_image is not None:
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            self.cv_image = cv2.transform(self.cv_image, kernel)
            self.cv_image = np.clip(self.cv_image, 0, 255).astype(np.uint8)
            self.display_image()

    def apply_vignette(self):
        if self.cv_image is not None:
            rows, cols = self.cv_image.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, 200)
            kernel_y = cv2.getGaussianKernel(rows, 200)
            kernel = kernel_y * kernel_x.T
            mask = 255 * kernel / np.linalg.norm(kernel)
            vignette = np.zeros_like(self.cv_image)
            for i in range(3):  # Apply vignette effect to each channel
                vignette[:, :, i] = self.cv_image[:, :, i] * mask
            self.cv_image = vignette.astype(np.uint8)
            self.display_image()

    def apply_blur(self):
        if self.cv_image is not None:
            self.cv_image = cv2.GaussianBlur(self.cv_image, (15, 15), 0)
            self.display_image()

    def apply_sharpen(self):
        if self.cv_image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.cv_image = cv2.filter2D(self.cv_image, -1, kernel)
            self.display_image()

    # Cropping, Resizing, Rotations
    def crop_image(self):
        if self.cv_image is not None:
            rows, cols = self.cv_image.shape[:2]
            self.cv_image = self.cv_image[rows // 4:rows * 3 // 4, cols // 4:cols * 3 // 4]
            self.display_image()

    def resize_image(self):
        if self.cv_image is not None:
            self.cv_image = cv2.resize(self.cv_image, (200, 200))
            self.display_image()

    def rotate_image(self, angle):
        if self.cv_image is not None:
            rows, cols = self.cv_image.shape[:2]
            matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            self.cv_image = cv2.warpAffine(self.cv_image, matrix, (cols, rows))
            self.display_image()

    def flip_horizontal(self):
        if self.cv_image is not None:
            self.cv_image = cv2.flip(self.cv_image, 1)
            self.display_image()

    def flip_vertical(self):
        if self.cv_image is not None:
            self.cv_image = cv2.flip(self.cv_image, 0)
            self.display_image()

# Run the app
root = tk.Tk()
app = ImageEditor(root)
root.mainloop()


# In[ ]:




