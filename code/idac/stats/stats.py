import numpy as np
import math
from datetime import datetime
from datetime import timedelta

class Stats:
    def __init__(self, config):
        self.species = config["classifier"]["species"]
        self.species.append('unknown')
        self.count = {}
        self.count['date'] = 0
        for spec in self.species:
            self.count[spec] = 0
        self.count['unknown'] = 0
        self.count['total'] = 0
        self.idhistory = {}
        self.details = {}

    def update_stats(self, oois, imname):
        print(imname)
        #Find info in filename
        ind = imname.index('-')
        startdate = imname[ind+1:ind+9]
        time = imname[ind+9:ind+15]
        time = ':'.join(time[i:i+2] for i in range(0, len(time), 2))

        for obj in oois:
            if obj.id in self.idhistory:
                obj.endtime = time
                self.calc_details(obj)
                self.idhistory[obj.id][0] += 1
                self.label_select(obj)
                if self.idhistory[obj.id][0] % 5 == 0:
                    #count down old
                    if self.idhistory[obj.id][2] != '':
                        self.count[self.idhistory[obj.id][2]] -= 1
                    #count up new
                    self.count[obj.label] += 1
                    if self.idhistory[obj.id][0] == 5:
                        self.count['total'] += 1
                    self.idhistory[obj.id][2] = obj.label
            else:
                self.idhistory[obj.id] = [1, np.zeros(len(self.species)), '']
                obj.starttime = time
                obj.endtime = time
                obj.startdate = startdate
                obj.count = 1
                self.calc_details(obj)
            arr = self.idhistory[obj.id][1]
            total_count = np.sum(arr[:8])
            total_count = total_count + arr[8] * 2
            obj.counts = total_count
            obj.boxsizehist.append(obj.w*obj.h)

    def label_select(self, obj):
        if obj.label == 'unknown':
            weight = 0.5
        else:
            weight = 1
        index = self.species.index(obj.label)
        self.idhistory[obj.id][1][index] += weight
        index = np.argmax(self.idhistory[obj.id][1])
        obj.label = self.species[index]

    def calc_details(self, obj):
        pos_iter = iter(obj.centerhist)
        prev_position = next(pos_iter)
        distance = 0
        if len(obj.centerhist) > 1:
            for pos in pos_iter:
                distance += int(math.sqrt(((pos[0] - prev_position[0]) ** 2) + ((pos[1] - prev_position[1]) ** 2)))
                prev_position = pos
        #print("Distance is: " + str(distance))
        self.details[obj.id] = obj
        obj.distance = distance

    def writedetails(self, dirname):
        file = open(dirname + '.json', 'w+')
        filecsv = open(dirname + '.csv', 'w+')
        line = 'id, startdate, starttime, endtime, duration, class, counts, confidence, size, distance\n'
        filecsv.write(line)
 
        for key in self.details.keys():
            obj = self.details[key]
            #Calculate confidence
            ind = self.species.index(obj.label)
            if obj.label == 'unknown' and obj.counts is not 0:
                conf = (self.idhistory[obj.id][1][ind]*2) / obj.counts
            elif obj.counts is not 0:
                conf = self.idhistory[obj.id][1][ind] / obj.counts

            #Distance
            distance = obj.distance

            #Calculate duration
            s1 = obj.starttime
            s2 = obj.endtime  # for example
            FMT = '%H:%M:%S'
            tdelta = datetime.strptime(s2, FMT) - datetime.strptime(s1, FMT)
            if tdelta.days < 0:
                tdelta = timedelta(days=0, seconds=tdelta.seconds, microseconds=tdelta.microseconds)
            tdelta_seconds = tdelta.total_seconds()

            #calculate avg blob size
            avg_blob = np.mean(obj.boxsizehist)

            #Format string
            if obj.counts > 3: #JBN??? should be same threshold as for statistic
                towrite = '{\'id\': ' + str(obj.id) + ', ' + '\'startdate\': ' + obj.startdate + ', ' + '\'starttime\': ' + obj.starttime + ', ' + '\'endtime\': ' + obj.endtime + ', ' \
                          + '\'duration\': ' + "%0.2f" % tdelta_seconds + ', ' + '\'class\': ' + '\'' + obj.label + '\', ' \
                          + '\'counts\': ' + str(obj.counts) + ', ' + '\'confidence\': ' + "%0.2f" % (conf*100) + ', ' + '\'size\': ' \
                          + "%0.2f" % avg_blob + ', ' + '\'distance\': ' + str(distance) + '},' + '\n'
    
                file.write(towrite)
                
                line = str(obj.id) + ', ' + obj.startdate + ', ' + obj.starttime + ', ' + obj.endtime + ', ' + "%0.2f" % tdelta_seconds + ', ' \
                       + obj.label + ', ' + str(obj.counts) + ', ' + "%0.2f" % (conf*100) + ', '  + "%0.2f" % avg_blob + ', ' + str(distance) + '\n'
                filecsv.write(line)

        file.close()
        filecsv.close()
        

