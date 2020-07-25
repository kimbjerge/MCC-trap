#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:22:55 2019

@author: jakob
"""


# TODO  centerhist

class ObjectOfInterrest:
    def __init__(self, x, y, w, h, id=0, centerhist=[]):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.centerx = int(x + w / 2)
        self.centery = int(y + h / 2)
        self.id = id
        self.centerhist = []
        self.label = ''
        self.percent = 0
        self.updatecenterhist()
        self.labelhist = []
        self.starttime = ''
        self.startdate = ''
        self.endtime = ''
        self.counts = 0
        self.boxsizehist = []
        self.distance = 0

    def updatecenterhist(self):
        self.centerx = int(self.x + self.w / 2)
        self.centery = int(self.y + self.h / 2)
        self.centerhist.append((self.centerx, self.centery))
