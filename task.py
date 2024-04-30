import sys
import random
from typing import List
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QEvent

class Point:
    def __init__(self, index, x, y, score):
        self.index = index
        self.x = x
        self.y = y
        self.score = score
    
    def __repr__(self):
        return f"Point({self.index}, {self.x}, {self.y}, {self.score})"

class PatternLearningTask(QWidget):
    def __init__(self, x_line_num, y_line_num):
        super().__init__()
        self.x_line_num = x_line_num
        self.y_line_num = y_line_num
        self.cell_size = 120 # Size of each cell
        self.width = None
        self.height = None
        self.points = None
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

    def paintEvent(self, event: QEvent) -> None:
        qp = QPainter()
        qp.begin(self)
        self.generateIntersectionPoints()
        self.drawGrid(qp)
        qp.end()

    def drawGrid(self, qp: QPainter) -> None:
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
    
    def generateIntersectionPoints(self) -> None:
        mid_x = self.width // 2
        mid_y = self.height // 2
        
        start_x = mid_x - (self.x_line_num - 1) // 2 * self.cell_size
        start_y = mid_y - (self.y_line_num - 1) // 2 * self.cell_size

        point_num = self.x_line_num * self.y_line_num

        # Generate list of points from 1 to point_num
        point_scores = list(range(1, point_num + 1))
        shuffled_point_scores = random.sample(point_scores, point_num) # Shuffle the list

        intersection_points: List[Point] = []
        index = 0
        for i in range(self.x_line_num):
            for j in range(self.y_line_num):
                x = start_x + i * self.cell_size
                y = start_y + j * self.cell_size
                score = shuffled_point_scores.pop() # Pop the last element from the list
                
                intersection_points.append(Point(index, x, y, score))
                index += 1
        
        self.points = intersection_points
        print(self.points)
        return 
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    x_line_num = 4
    y_line_num = 4
    task = PatternLearningTask(x_line_num, y_line_num)
    sys.exit(app.exec_())