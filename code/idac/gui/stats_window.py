from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import PyQt5

class StatsWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.table = QTableWidget()
        self.table.setRowCount(12)
        self.table.setColumnCount(2)
        self.setGeometry(1420, 610, 250, 400)
        self.table.setItem(12, 0, QTableWidgetItem("Total"))
        self.table.setEditTriggers(PyQt5.QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setSelectionMode(PyQt5.QtWidgets.QAbstractItemView.NoSelection)
        self.setWindowTitle('Stats')
        self.setCentralWidget(self.table)