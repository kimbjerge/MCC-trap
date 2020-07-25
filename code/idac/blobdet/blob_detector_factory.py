from idac.blobdet.adaptive_blobdetector import AdaptiveBlobDetector
from idac.blobdet.simplethres_blobdetector import SimpleThresBlobDetector
from idac.blobdet.otsu_blobdetector import OtsuBlobDetector
from idac.blobdet.custom_blobdetector import CustomBlobDetector


class BlobDetectorFactory():

    @staticmethod
    def get_blob_detector(type, config):
        try:
            if type == 'adaptive':
                return AdaptiveBlobDetector(config)
            if type == 'simple':
                return SimpleThresBlobDetector(config)
            if type == 'otsu':
                return OtsuBlobDetector(config)
            if type == 'custom':
                return CustomBlobDetector(config)
                

            raise AssertionError('The blob detector does not exist!')
        except AssertionError as e:
            print(e)

