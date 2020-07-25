import cv2
from pathlib import Path


class MovieMaker:
    def __init__(self, config, name='Result.avi'):
        config = config['moviemaker']
        self.maxframes = config['maxframes']
        self.maxim = config['maximages']
        self.resultdir = config['resultdir']
        self.writeim = config['writeimages']
        self.writemovie = config['writemovie']
        self.imcounter = 0
        self.framecounter = 0
        self.size = (3840, 2160)
        self.fps = config['fps']
        finaldir = Path(self.resultdir) / name
        if self.writemovie:
            self.writer = cv2.VideoWriter(str(finaldir), cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.size)

    def getImDateTime(self, imname):
        
        ind = imname.index('-')
        imdate = imname[ind+1:ind+9]
        imtime = imname[ind+9:ind+15]
        imtime = ':'.join(imtime[i:i+2] for i in range(0, len(imtime), 2))
        
        return imdate, imtime

    def writeframe(self, im, imname=''):

        if (self.writemovie and self.maxframes >= self.framecounter) or (self.writemovie and self.maxframes == -1):
            if imname != '':
                imdate, imtime = self.getImDateTime(imname)
                dateTime = imdate + ' ' + imtime
                cv2.putText(im, dateTime, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            self.writer.write(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            self.framecounter = self.framecounter + 1

        if (self.writeim and self.maxim >= self.imcounter) or (self.writeim and self.maxim == -1):
            cv2.imwrite(self.resultdir + '/' + str(self.imcounter) + '.jpg', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            self.imcounter = self.imcounter + 1

        return 0

    def releasemovie(self):
        self.writer.release()
        print('Movie released!')
        print('Frames written: ' + str(self.framecounter))
        print('Images written: ' + str(self.imcounter))
        return 0
