from PyQt5.QtCore import QThread, pyqtSignal
import time
from PyQt5.QtGui import QImage
from idac.datareader.data_reader import DataReader
from idac.tracker.tracker import Tracker
from idac.blobdet.blob_detector_factory import BlobDetectorFactory
from idac.configreader.configreader import readconfig
from idac.imagemod.image_mod import Imagemod
from idac.moviemaker.movie_maker import MovieMaker
from idac.classifier.classifier_factory import ClassifierFactory
from idac.stats.stats import Stats
from pathlib import Path
import os


class IdacRunner(QThread):
    imagesignal = pyqtSignal('PyQt_PyObject')
    binarysignal = pyqtSignal('PyQt_PyObject')
    statsignal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        QThread.__init__(self)
        self.pause = False

    def run(self):
        base_path = Path(__file__).parent.parent.parent.parent
        config_filename = os.path.join(base_path, 'config/MCC_config.json')
        conf = readconfig(config_filename)
        writemovie = conf['moviemaker']['writemovie']
        reader = DataReader(conf)
        gen = reader.getimage()
        print(type(gen))
        bl = BlobDetectorFactory.get_blob_detector(conf['blobdetector']['type'], conf)
        tr = Tracker(conf)
        imod = Imagemod()
        mm = MovieMaker(conf)
        clas = ClassifierFactory.get_classifier(conf['classifier']['type'], conf)
        stat = Stats(conf)

        im, file = gen.__next__()
        startid = 0
        image_new, count, ooi1, id, binary = bl.findboxes(im, startid)
        for oi in ooi1:
            oi.id = startid
            startid = startid + 1

        clas.makeprediction(im, ooi1)

        for im, file in gen:
            time1 = time.time()
            goods, startid, binary = tr.track_frames(ooi1, im, startid)
            height, width = binary.shape
            binary = binary
            bytesPerLine = width
            qImg = QImage(binary.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            # Prepare for ui change
            self.binarysignal.emit(qImg)
            ooi1 = goods
            clas.makeprediction(im, goods)
            stat.update_stats(goods, file)
            image = imod.drawoois(im, goods)
            # Prepare for ui change
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            while self.pause:
                self.sleep(1)
            self.imagesignal.emit(qImg)
            self.statsignal.emit(stat.count)

            # Write frame
            mm.writeframe(image, file)
            time2 = time.time()
            print('function took {:.3f} ms'.format((time2 - time1) * 1000.0))

        if writemovie:
            mm.releasemovie()
