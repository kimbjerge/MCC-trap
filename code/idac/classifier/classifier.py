from abc import ABC, abstractmethod


class Classifier(ABC):

    @abstractmethod
    def makeprediction(self, img, ooi):
        pass
