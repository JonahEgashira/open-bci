import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt

class PatternLearningTask(QWidget):
    def __init__(self, x_line_num, y_line_num):
        super().__init__()
        self.x_line_num = x_line_num
        self.y_line_num = y_line_num
        self.cell_size = 120 # Size of each cell
        self.width = None
        self.height = None
        self.initUI()

    def initUI(self):
        self.showFullScreen()
        self.setWindowTitle('Pattern Learning Task')
        self.setStyleSheet("background-color: white;")  # Set background color to white

        screen = self.screen()
        size = screen.size()
        
        self.width = size.width()
        self.height = size.height()

        print(f"Screen width: {self.width}, Screen height: {self.height}")

        self.show()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.drawGrid(qp)
        qp.end()

    def drawGrid(self, qp):
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)

        mid_x = self.width // 2
        mid_y = self.height // 2
        
        start_x = mid_x - (self.x_line_num - 1) // 2 * self.cell_size
        start_y = mid_y - (self.y_line_num - 1) // 2 * self.cell_size
        
        for i in range(self.x_line_num):
            sx = start_x + i * self.cell_size # 線のX座標始点
            sy = start_y - self.cell_size # 線のY座標始点
            gx = start_x + i * self.cell_size # 線のX座標終点
            gy = start_y + self.cell_size * self.y_line_num # 線のY座標終点

            qp.drawLine(sx, sy, gx, gy)
        
        for i in range(self.y_line_num):
            sx = start_x - self.cell_size
            sy = start_y + i * self.cell_size
            gx = start_x + self.cell_size * self.x_line_num
            gy = start_y + i * self.cell_size
            
            qp.drawLine(sx, sy, gx, gy)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    x_line_num = 4
    y_line_num = 4
    task = PatternLearningTask(x_line_num, y_line_num)
    sys.exit(app.exec_())