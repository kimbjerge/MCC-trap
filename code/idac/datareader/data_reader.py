"""
Created on Mon Sep 16 15:15:36 2019

@author: jakob
"""
import os
from skimage import io
from pathlib import Path


class DataReader:
    def __init__(self, config):
        print('config read')
        self.data_path = config['datareader']['datapath']
        self.maximages = config['datareader']['maxim']


    def getimage(self):
        total = len(os.listdir(self.data_path))
        print('Found ' + str(total) + ' images.')
        print('Max images: ' + str(self.maximages))
        counter = 0
        for file in sorted(os.listdir(self.data_path)):
            if file.endswith('.jpg'):
                file_name = Path(self.data_path) / file
                img = io.imread(file_name)
                print(file_name)
                yield img, file
                if self.maximages == counter:
                    break
                counter = counter + 1

        print("Read: " + str(counter) + " OF " + str(self.maximages))
