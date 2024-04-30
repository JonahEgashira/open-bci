import sys
import random
from typing import List
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QEvent


class Point:
    def __init__(self, index, x, y, score, trials=50):
        self.index = index
        self.x = x
        self.y = y
        self.score = score
        self.trials = trials
        self.trial_count = 0

    def __repr__(self):
        return f"Point({self.index}, {self.x}, {self.y}, {self.score})"


class PatternLearningTask(QWidget):
    def __init__(self, x_line_num, y_line_num):
        super().__init__()
        self.x_line_num = x_line_num
        self.y_line_num = y_line_num
        self.point_num = x_line_num * y_line_num
        self.cell_size = 120  # Size of each cell
        self.width = None
        self.height = None
        self.points: List[Point] = []
        self.selected_points: List[Point] = []
        self.initUI()

    def initUI(self) -> None:
        self.showFullScreen()
        self.setWindowTitle("Pattern Learning Task")
        self.setStyleSheet("background-color: white;")  # Set background color to white

        screen = self.screen()
        size = screen.size()

        self.width = size.width()
        self.height = size.height()

        self.result_label = QLabel(self)
        label_width = 200
        label_height = 100
        self.result_label.setGeometry(self.width // 2, self.height // 8, label_width, label_height)
        self.result_label.setStyleSheet("font-size: 36px; color: black;")
        self.result_label.setText("Start")
        self.result_label.show()

        self.generateIntersectionPoints()

        print(f"Screen width: {self.width}, Screen height: {self.height}")

        self.show()

        self.selectRandomTwoPoints()
        return

    def keyPressEvent(self, event: QEvent) -> None:
        user_input = event.key()
        
        print(f"User input: {user_input}")
        
        isBelowAverage = self.isSelectedPointsBelowAverage()
        correct = False

        if user_input == Qt.Key_L and isBelowAverage:
            correct = True
        elif user_input == Qt.Key_O and (not isBelowAverage):
            correct = True

        status_text = "Correct" if correct else "Incorrect"
        
        print(status_text)

        self.result_label.setText(status_text)
        self.result_label.show()

        self.update()

        return

    def isSelectedPointsBelowAverage(self) -> bool:
        average = (1 + self.point_num) / 2
        score_sum = sum([point.score for point in self.selected_points])

        return score_sum < average

    def paintEvent(self, event: QEvent) -> None:
        qp = QPainter()
        qp.begin(self)
        self.drawGrid(qp)
        self.drawSelectedPoints(qp)
        qp.end()
        return

    def drawSelectedPoints(self, qp: QPainter) -> None:
        point_size = 16
        pen = QPen(Qt.red, point_size, Qt.SolidLine)
        qp.setPen(pen)

        for point in self.selected_points:
            qp.drawPoint(point.x, point.y)

    def drawGrid(self, qp: QPainter) -> None:
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)

        mid_x = self.width // 2
        mid_y = self.height // 2

        start_x = mid_x - (self.x_line_num - 1) // 2 * self.cell_size
        start_y = mid_y - (self.y_line_num - 1) // 2 * self.cell_size

        for i in range(self.x_line_num):
            sx = start_x + i * self.cell_size  # 線のX座標始点
            sy = start_y - self.cell_size  # 線のY座標始点
            gx = start_x + i * self.cell_size  # 線のX座標終点
            gy = start_y + self.cell_size * self.y_line_num  # 線のY座標終点

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
        shuffled_point_scores = random.sample(
            point_scores, point_num
        )  # Shuffle the list

        intersection_points: List[Point] = []
        index = 0
        for i in range(self.x_line_num):
            for j in range(self.y_line_num):
                x = start_x + i * self.cell_size
                y = start_y + j * self.cell_size
                score = (
                    shuffled_point_scores.pop()
                )  # Pop the last element from the list

                intersection_points.append(Point(index, x, y, score))
                index += 1

        self.points = intersection_points
        print(self.points)
        return

    def selectRandomTwoPoints(self) -> None:
        selected_points = random.sample(self.points, 2)
        self.selected_points = selected_points
        print(self.selected_points)
        self.update()
        return


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x_line_num = 4
    y_line_num = 4
    task = PatternLearningTask(x_line_num, y_line_num)
    sys.exit(app.exec_())
