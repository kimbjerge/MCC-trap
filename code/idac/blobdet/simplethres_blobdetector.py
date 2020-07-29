import cv2
import numpy as np
from idac.objectOfInterrest import ObjectOfInterrest
from idac.blobdet.blob_detector import Blobdetector

#Reference simple - https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
#Last accessed 17/12 - 2019

class SimpleThresBlobDetector(Blobdetector):
    def __init__(self, config):
        self.config = config['blobdetector']
        background = cv2.imread(self.config['backgroundpath'])
        self.background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        self.kernel_fast = self.config["simple"]["kernel"]
        self.thresh = self.config["simple"]["thresh"]
        self.minarea = self.config["minarea"]
        self.maxarea = self.config["maxarea"]


    def findboxes(self, img, startingid):
        original = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray - self.background
        gray[gray > self.thresh] = 255
        gray[gray <= self.thresh] = 0
        kernel = np.ones((self.kernel_fast*4, self.kernel_fast*4), np.uint8) #KBE improved ???
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((self.kernel_fast, self.kernel_fast), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        binary = gray.copy()
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
                # TO DO add to objects of interest
                obj = ObjectOfInterrest(x, y, w, h)
                ooi.append(obj)
                count = count + 1

        return original, count, ooi, startingid, binary
