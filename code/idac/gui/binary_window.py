from PyQt5.QtWidgets import QMainWindow, QLabel


class BinaryWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.label = QLabel()
        self.label.setText('Second')
        self.label.setScaledContents(1)
        self.setCentralWidget(self.label)
        self.setWindowTitle('Binary view')
