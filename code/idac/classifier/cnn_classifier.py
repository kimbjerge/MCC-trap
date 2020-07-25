import tensorflow as tf
import cv2
import numpy as np
from idac.classifier.classifier import Classifier
from skimage import io
from skimage import color
from skimage.transform import resize
from PIL import Image
import scipy
from pathlib import Path

class CnnClassifier(Classifier):
    def __init__(self, conf):
        self.config = conf["classifier"]
        self.modeltype = self.config["modeltype"]
        self.model = self.loadmodel()
        self.species = self.config["species"]
        self.imgdim = self.config["imagedimension"]
        self.uknown_dir = self.config["unknown_dir"]
        self.write_unknown = self.config["write_unknown"]
        self.padding = self.config["boundingboxpadding"]
 
        self.threshold_unknown = self.config["threshold_unknown"]
        self.counter = 0

    def loadmodel(self):
        model = tf.keras.models.load_model(self.modeltype + '.h5')
        print('Loaded model: ', self.modeltype + '.h5')
        model.summary()
        return model

    def cropimg(self, img, ooi):
        x = ooi.x
        y = ooi.y
        w = ooi.w
        h = ooi.h
        crop_padding = self.padding # Version 2 128x128 103, Version 1 64x64 135
        w1 = h1 = 470 # Version 2 change to 360 average dataset crop size, not used
        newh = (h1 - h) // 2
        neww = (w1 - w) // 2
        if (h1 - h) % 2 != 0:
            h = h + 1
        if (w1 - w) % 2 != 0:
            w = w + 1

        padding_method = True

        if not padding_method:
            if y > newh and x > neww:
                img_crop = img[y - newh: y + h + newh, x - neww:x + w + neww]
            else:
                img_crop = img[y: y + h, x:x + w]
        else:
            if y > crop_padding and x > crop_padding:
                img_crop = img[y-crop_padding: y + h+crop_padding, x-crop_padding:x + w+crop_padding]
            else:
                img_crop = img[y: y + h, x:x + w]
        dim = (self.imgdim, self.imgdim)
        method_pil = False
        if method_pil:
            img224 = Image.fromarray(img_crop)
            img224 = img224.resize(dim)
            resized = np.array(img224)
            resized = resized*1./255
        else:
            img224 = resize(img_crop, dim, mode='reflect')
            resized = img224
        return resized

    def makeprediction(self, img, ooi):
        images = []
        for OI in ooi:
            temp = self.cropimg(img, OI)
            if temp.shape == (self.imgdim, self.imgdim, 3):
                images.append(temp)
        to_pred = np.array(images).astype(np.float32)
        if len(to_pred != 0):
            predictions = self.model.predict_proba(to_pred, verbose=1)
            for i in range(len(ooi)):
                probability = np.amax(predictions[i])
                index = np.where(predictions[i] == probability)
                ooi[i].label = self.species[index[0][0]]
                if probability * 100 > self.threshold_unknown:
                    ooi[i].percent = probability * 100
                    #if ooi[i].y > 1600:
                    #cv2.imwrite(self.uknown_dir + 'Unknown' + str(self.counter) + '.jpg', cv2.cvtColor(images[i] * 255, cv2.COLOR_RGB2BGR))
                        #io.imsave(self.uknown_dir + 'Unknown' + str(self.counter) + '.jpg', images[i] * 255)
                    self.counter += 1
                else:
                    ooi[i].label = 'unknown'
                    ooi[i].percent = -1
                    if self.write_unknown:
                        file_name = Path(self.uknown_dir) / str('Unknown' + str(self.counter) + '.jpg')
                        io.imsave(file_name, images[i])
                    self.counter += 1
        return ooi, img
