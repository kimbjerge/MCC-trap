from PyQt5.QtWidgets import QMainWindow, QLabel
from PyQt5.QtWidgets import QProgressBar, QVBoxLayout, QWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
from idac.gui.binary_window import BinaryWindow
from idac.gui.idac_runner import IdacRunner
from idac.gui.stats_window import StatsWindow
from idac.gui.view_tree import ViewTree
from idac.configreader.configreader import readconfig
from PyQt5 import QtCore
import os
from pympler.tracker import SummaryTracker
from pathlib import Path


tracker = SummaryTracker()

from PyQt5.QtGui import QIcon, QPixmap


class IdacMain(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.runner = IdacRunner()
        base_path = Path(__file__).parent.parent.parent.parent
        config_filename = os.path.join(base_path, 'config/MCC_config.json')
        self.conf = readconfig(config_filename)
        self.maximages = self.conf['datareader']['maxim']
        if self.maximages < 0:
            self.total = len(os.listdir(self.conf['datareader']['datapath']))
        else:
            self.total = self.maximages + 1
        self.counter = 1
        self.statusBar()
        self.label = QLabel()
        self.tree = ViewTree(self.conf)
        self.tree.header().hide()
        self.tree.setFocusPolicy(Qt.NoFocus)
        self.tree.collapseAll()
        self.label.setText('Press "S" To start')
        self.runner.imagesignal.connect(self.update)
        self.setGeometry(300, 300, 1920, 1080)
        self.setWindowTitle('Main window')
        grid = QVBoxLayout()
        grid.addWidget(self.label)
        grid.addWidget(QLabel('Current config:'))
        grid.addWidget(self.tree)
        self.widget = QWidget()
        self.widget.setLayout(grid)
        self.setCentralWidget(self.widget)
        self.progressBar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progressBar)
        self.statusBar().showMessage('READY')
        self.progressBar.setMaximum(100)
        self.setWindowIcon(QIcon('icon.png'))
        self.setWindowTitle('Insect Detection And Classification')
        self.setWindowIconText('IDAC')
        self.secondWindow = BinaryWindow()
        self.runner.binarysignal.connect(self.updateBinary)
        self.runner.statsignal.connect(self.update_statwindow)
        self.statsw = StatsWindow()

        self.show()

    def update(self, event):
        self.counter = self.counter + 1
        self.progressBar.setValue(int((self.counter / self.total) * 100) + 2)
        self.label.setPixmap(QPixmap(event).scaled(self.label.width(), self.label.height(), QtCore.Qt.KeepAspectRatio))
        self.label.adjustSize()
        self.statusBar().showMessage('RUNNING... Press C to terminate          Image: ' + str(self.counter) + ' OF '
                                     + str(self.total))

    def updateBinary(self, event):
        self.secondWindow.label.setPixmap(QPixmap(event).scaled(self.secondWindow.label.width(),
                                                                self.secondWindow.label.height(),
                                                                QtCore.Qt.KeepAspectRatio))
        self.secondWindow.label.adjustSize()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
            self.runner.pause = not self.runner.pause
            self.statusBar().showMessage('PAUSED')
        elif event.key() == QtCore.Qt.Key_S:
            self.setCentralWidget(self.label)
            self.runner.start()
            self.label.setText('STARTING NOW. Please wait.....')
            self.statusBar().showMessage('RUNNING... Press C to cancel')
        elif event.key() == QtCore.Qt.Key_C:
            self.runner.terminate()
            self.secondWindow.close()
            self.statsw.close()
            self.close()
        elif event.key() == QtCore.Qt.Key_B:
            self.secondWindow.show()
        elif event.key() == QtCore.Qt.Key_D:
            tracker.print_diff()
        elif event.key() == QtCore.Qt.Key_T:
            if self.statsw.isVisible():
                self.statsw.close()
            else:
                self.statsw.show()

    def update_statwindow(self, event):
        row_count = 0
        for obj in event:
            self.statsw.table.setItem(row_count, 0, QTableWidgetItem(obj))
            self.statsw.table.setItem(row_count, 1, QTableWidgetItem(str(event[obj])))
            row_count += 1
