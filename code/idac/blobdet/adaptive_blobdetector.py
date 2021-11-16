import cv2
import numpy as np
from idac.objectOfInterrest import ObjectOfInterrest
from idac.blobdet.blob_detector import Blobdetector
import time

#Reference adaptive - https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#Last accessed 17/12 - 2019

class AdaptiveBlobDetector(Blobdetector):
    def __init__(self, config):
        self.config = config['blobdetector']
        background = cv2.imread(self.config['backgroundpath'])
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        self.usebackground = self.config["usebackground"]
        self.minarea = self.config["minarea"]
        self.maxarea = self.config["maxarea"]
        self.adapsize = self.config["adaptive"]["adapsize"]
        self.adapconst = self.config["adaptive"]["adapconst"]
        self.medianblur = self.config["adaptive"]["medianblur"]
        self.kernel_adap = self.config["adaptive"]["kernel"]

    def findboxes(self, img, startingid):
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if self.usebackground: # Substract background image
            gray = gray - self.background
            
        gray = cv2.medianBlur(gray, self.medianblur)
            
        time1 = time.time()
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, self.adapsize, self.adapconst)
        time2 = time.time()
        print('Adaptive took {:.3f} ms'.format((time2 - time1) * 1000.0))
        kernel = np.ones((self.kernel_adap, self.kernel_adap), np.uint8)
        gray_closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        original = cv2.morphologyEx(gray_closed, cv2.MORPH_OPEN, kernel)
        binary = original.copy()
        ooi = []
        startingid = startingid
        # Init counters for counting number of images
        count = 0
        _, contours, _= cv2.findContours(255 - binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            crop_img = original[y:y + h, x:x + w]
            areaSize = crop_img.shape[0] * crop_img.shape[1]
            if areaSize > self.minarea and areaSize < self.maxarea:
                obj = ObjectOfInterrest(x, y, w, h)
                ooi.append(obj)
                count = count + 1

        return original, count, ooi, startingid, binary
