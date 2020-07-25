from idac.classifier.cnn_classifier import CnnClassifier

class ClassifierFactory:

    @staticmethod
    def get_classifier(type, conf):
        try:
            if type == 'cnn':
                return CnnClassifier(conf)

            raise AssertionError('The classifier does not exist!')
        except AssertionError as e:
            print(e)
