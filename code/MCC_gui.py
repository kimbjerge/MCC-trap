import sys
from PyQt5.QtWidgets import QApplication
from idac.gui.idac_main import IdacMain

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = IdacMain()
    sys.exit(app.exec_())
