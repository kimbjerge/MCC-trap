"""
Created on Fri Sep 13 15:15:36 2019
@author: jakob
"""
from abc import ABC, abstractmethod


# TODO -> ret parametere og metode ift jupyter notebooken -> done

class Blobdetector(ABC):

    @abstractmethod
    def findboxes(self, img, startingid):
        pass
