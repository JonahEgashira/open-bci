import sys
import random
from datetime import datetime
from typing import List
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QEvent, QTimer


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


class ReactionInfo:
    def __init__(self, trial_num, key_press_time, is_correct):
        self.trial_num = trial_num
        self.key_press_time = key_press_time
        self.is_correct = is_correct


class PatternLearningTask(QWidget):
    def __init__(self, x_line_num, y_line_num, trials=50):
        super().__init__()
        self.x_line_num = x_line_num
        self.y_line_num = y_line_num
        self.trials = trials
        self.trial_count = 0
        self.point_num = x_line_num * y_line_num
        self.cell_size = 120  # Size of each cell
        self.width = None
        self.height = None
        self.reactions: List[ReactionInfo] = []
        self.waiting_for_start = True
        self.key_event_enabled = False
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

        print(f"Screen width: {self.width}, Screen height: {self.height}")

        self.generateIntersectionPoints()
        self.prepareLabel()
        return

    def prepareLabel(self) -> None:
        self.result_label = QLabel(self)
        self.result_label.setText("Press Any Key")
        label_width = 200
        label_height = 60

        self.result_label.setGeometry(
            (self.width - label_width) // 2,
            (self.height - label_height) // 12,
            label_width,
            label_height,
        )
        self.result_label.setStyleSheet(
            "font-size: 20px; color: black; background-color: #f0f0f0;"
        )
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.show()
        return

    def startTask(self):
        self.waiting_for_start = False
        self.selectRandomTwoPoints()

    def keyPressEvent(self, event: QEvent) -> None:
        if self.waiting_for_start:
            self.result_label.setText("Starting...")
            self.result_label.show()
            QTimer.singleShot(1500, self.startTask)
            self.waiting_for_start = False
            return

        if not self.key_event_enabled:
            return

        user_input = event.key()

        print(f"User input: {user_input}")

        if user_input in [Qt.Key_L, Qt.Key_O]:
            key_press_time = datetime.now()
            self.key_event_enabled = False
            isBelowAverage = self.isSelectedPointsBelowAverage()

            is_correct = (user_input == Qt.Key_L and isBelowAverage) or (
                user_input == Qt.Key_O and not isBelowAverage
            )

            reaction = ReactionInfo(self.trial_count, key_press_time, is_correct)
            self.reactions.append(reaction)

            status_text = "Correct" if is_correct else "Incorrect"

            self.result_label.setText(status_text)
            self.result_label.show()

            self.trial_count += 1

            if self.trial_count > self.trials:
                QMessageBox.information(self, "Information", "Task is finished")
                self.close()
            else:
                wait_time = 1500  # 1.5 seconds
                QTimer.singleShot(wait_time, self.selectRandomTwoPoints)

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
        if not self.key_event_enabled:
            return

        radius = 16
        pen_width = 4
        pen = QPen(Qt.red, pen_width)
        qp.setPen(pen)
        qp.setBrush(Qt.red)

        for point in self.selected_points:
            qp.drawEllipse(point.x - radius, point.y - radius, 2 * radius, 2 * radius)

    def drawGrid(self, qp: QPainter) -> None:
        pen = QPen(Qt.black, 4, Qt.SolidLine)
        qp.setPen(pen)

        mid_x = self.width // 2
        mid_y = self.height // 2

        start_x = int(mid_x - (self.x_line_num - 1) * self.cell_size / 2)
        start_y = int(mid_y - (self.y_line_num - 1) * self.cell_size / 2)

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

        start_x = int(mid_x - (self.x_line_num - 1) * self.cell_size / 2)
        start_y = int(mid_y - (self.y_line_num - 1) * self.cell_size / 2)

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
        self.result_label.hide()
        selected_points = random.sample(self.points, 2)
        self.key_event_enabled = True
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
