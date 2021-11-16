import cv2
import numpy as np
from idac.objectOfInterrest import ObjectOfInterrest
from idac.blobdet.blob_detector import Blobdetector

#Reference otsu - https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#Last accessed 17/12 - 2019

class OtsuBlobDetector(Blobdetector):
    def __init__(self, config):
        self.config = config['blobdetector']
        background = cv2.imread(self.config['backgroundpath'])
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        self.usebackground = self.config["usebackground"]
        self.minarea = self.config["minarea"]
        self.maxarea = self.config["maxarea"]
        self.blur_kernel = self.config["otsu"]["blur_kernel"]
        self.blur_const = self.config["otsu"]["blur"]
        self.kernel_otsu = self.config["otsu"]["morph_kernel"]

    def findboxes(self, img, startingid):
        original = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if self.usebackground:
            gray = gray - self.background
        blur = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), self.blur_const)
        th, image_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((self.kernel_otsu, self.kernel_otsu), np.uint8)
        gray = cv2.morphologyEx(image_otsu, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        binary = gray.copy()
        ooi = []
        startingid = startingid
        # Init counters for counting number of images
        count = 0
        _, contours, _= cv2.findContours(255-binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            crop_img = original[y:y + h, x:x + w]
            areaSize = crop_img.shape[0] * crop_img.shape[1]
            if areaSize > self.minarea and areaSize < self.maxarea:
                obj = ObjectOfInterrest(x, y, w, h)
                ooi.append(obj)
                count = count + 1

        return original, count, ooi, startingid, binary