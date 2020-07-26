# -*- coding: utf-8 -*-
"""
Created on Mon May  4 08:10:16 2020

@author: Kim Bjerge
"""
import os
import time
from pathlib import Path
from idac.configreader.configreader import readconfig
from idac.datareader.data_reader import DataReader
from idac.tracker.tracker import Tracker
from idac.blobdet.blob_detector_factory import BlobDetectorFactory
from idac.imagemod.image_mod import Imagemod
from idac.moviemaker.movie_maker import MovieMaker
from idac.classifier.classifier_factory import ClassifierFactory
from idac.stats.stats import Stats
from PyQt5.QtGui import QImage
       
def run(dirName):
    config_filename = '../config/MCC_config.json'
    conf = readconfig(config_filename)
    conf['datareader']['datapath'] += '/' + dirName
    print(conf['datareader']['datapath'])
    total = len(os.listdir(conf['datareader']['datapath']))
    print(conf['moviemaker']['resultdir'])
    writemovie = conf['moviemaker']['writemovie']
    reader = DataReader(conf)
    gen = reader.getimage()
    print(type(gen))
    bl = BlobDetectorFactory.get_blob_detector(conf['blobdetector']['type'], conf)
    tr = Tracker(conf)
    imod = Imagemod()
    if dirName == '':
        dirName = 'tracks'
    mm = MovieMaker(conf, name=dirName + '.avi')
    clas = ClassifierFactory.get_classifier(conf['classifier']['type'], conf)
    stat = Stats(conf)

    im, file = gen.__next__()
    startid = 0
    image_new, count, ooi1, id, binary = bl.findboxes(im, startid)
    for oi in ooi1:
        oi.id = startid
        startid = startid + 1

    clas.makeprediction(im, ooi1)

    iterCount = 0
    for im, file in gen:
        iterCount += 1
        print('Image nr. ' + str(iterCount) + '/' + str(total))
        time1 = time.time()
        goods, startid, binary = tr.track_frames(ooi1, im, startid)
        height, width = binary.shape
        binary = binary
        bytesPerLine = width
        qImg = QImage(binary.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
        ooi1 = goods
        clas.makeprediction(im, goods)
        stat.update_stats(goods, file)
        print(stat.count)
        image = imod.drawoois(im, goods)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)

        # Write frame
        mm.writeframe(image, file)
        time2 = time.time()
        print('Processing image took {:.3f} ms'.format((time2 - time1) * 1000.0))

    if writemovie:
        mm.releasemovie()
        
    resultdir = conf['moviemaker']['resultdir'] + '/'
    stat.writedetails(resultdir + dirName)

    return stat, resultdir


def print_totals(date, stat, resultdir):
    record = str(date) + ','
    for spec in stat.species:
        print(spec, stat.count[spec])
        record += str(stat.count[spec]) + ','
    print('Total', stat.count['total'])
    record += str(stat.count['total']) + '\n'

    file = open(resultdir + 'statistics.csv', 'a')
    file.write(record)
    file.close()

    stat.count['date'] = date
    file = open(resultdir + 'statistics.json', 'a')
    file.write(str(stat.count) + '\n')
    file.close()


if __name__ == '__main__':

    print('STARTING NOW. Please wait.....')
    #dirNames = ['3108_Brio', '0109_Brio', '0209_Brio']  # Sub directories in data folder - not included in github
    #dirNames = [''] # Use images in data directory
    dirNames =['2608_data'] # Sub directory with some other sample images
    for dirName in dirNames:
        print(dirName)
        stat, resultdir = run(dirName)
        if dirName == '':
            date = 901
        else:    
            date = int(dirName[2:4]) * 100 + int(dirName[0:2])  # Convert to format MMDD
        print_totals(date, stat, resultdir)
