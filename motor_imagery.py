import argparse
import time
import sys
import random
import threading
import numpy as np
import pandas as pd
import logging
import os

import brainflow
from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)
from brainflow.data_filter import (
    DataFilter,
    FilterTypes,
    AggOperations,
    DetrendOperations,
)

from datetime import datetime
from typing import List
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QEvent, QTimer


DEFAULT_CHANNELS = [7] # C3 channel

class MotorImageryTask(QWidget):
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.directory_handler = DirectoryHandler()
        self.eeg_handler = EEGHandler(self.board, self.directory_handler, DEFAULT_CHANNELS)
        self.eeg_thread = threading.Thread(target=self.eeg_handler.collect_data)

        self.trials = 3
        self.trial_count = 0
        self.wait_time = 5000  # 5000 ms
        self.width = None
        self.height = None
        self.waiting_for_start = True
        self.initUI()

    def initUI(self) -> None:
        self.showFullScreen()
        self.setWindowTitle("Motor Imagery Task")
        self.setStyleSheet("background-color: white;")
        self.initScreen()
        self.prepareLabel()
        return

    def initScreen(self) -> None:
        screen = self.screen()
        size = screen.size()
        self.width = size.width()
        self.height = size.height()
        print(f"Screen width: {self.width}, Screen height: {self.height}")
        return

    def closeEvent(self, event):
        logging.info("Closing the application")
        self.eeg_handler.stop()
        self.eeg_thread.join()
        super().closeEvent(event)

    def prepareLabel(self) -> None:
        self.instruction_label = QLabel(self)
        self.instruction_label.setText("Press Any Key")
        label_width = 400
        label_height = 60

        self.instruction_label.setGeometry(
            (self.width - label_width) // 2,
            (self.height - label_height) // 2,
            label_width,
            label_height,
        )
        self.instruction_label.setStyleSheet(
            "font-size: 20px; color: black; background-color: #f0f0f0;"
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.show()
        return

    def startTask(self):
        self.waiting_for_start = False
        self.eeg_thread.start()
        self.showInstruction()
        self.directory_handler.create_directory()
        return

    def keyPressEvent(self, event: QEvent) -> None:
        if self.waiting_for_start:
            self.instruction_label.setText("Starting...")
            self.instruction_label.show()
            QTimer.singleShot(self.wait_time, self.startTask)
            self.waiting_for_start = False
            return

    def showInstruction(self):
        if self.trial_count >= self.trials:
            self.end_task()
            return

        if self.trial_count % 2 == 0:
            instruction = "Imagine grasping your right hand"
            label = 1
        else:
            instruction = "Rest"
            label = 0

        self.eeg_handler.set_label(label)
        self.instruction_label.setText(instruction)
        self.instruction_label.show()
        self.trial_count += 1

        QTimer.singleShot(self.wait_time, self.showInstruction)

    def end_task(self):
        QMessageBox.information(self, "Information", "Task Completed")
        self.close()


class DirectoryHandler:
    def __init__(self):
        self.time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_directory(self):
        if not os.path.exists("./data"):
            os.makedirs("./data")

        if not os.path.exists(f"./data/{self.time_stamp}"):
            os.makedirs(f"./data/{self.time_stamp}")

    def get_directory_path(self):
        return f"./data/{self.time_stamp}"


class EEGHandler:
    def __init__(self, board, directory_handler, channels=None):
        self.board = board
        self.stop_signal = False
        self.directory_handler = directory_handler
        self.start_time = None
        self.current_label = None
        if channels is None:
            self.channels = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.channels = channels
    
    def set_label(self, label):
        self.current_label = label

    def collect_data(self):
        try:
            self.board.prepare_session()
            self.board.start_stream()
            self.start_time = time.time()  # データ収集の開始時刻を記録

            print("Start streaming")
            print("Sfreq: ", self.board.get_sampling_rate(BoardIds.CYTON_BOARD))

            data_dir = "./data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            directory_path = self.directory_handler.get_directory_path()
            file_path = f"{directory_path}/eeg_data.csv"

            eeg_data_list = []
            timestamp_list = []
            label_list = []

            while not self.stop_signal:
                data = self.board.get_board_data()
                eeg_channels = self.board.get_eeg_channels(self.board.board_id)
                selected_channels = [eeg_channels[c - 1] for c in self.channels]  # 指定されたチャンネルに対応するインデックスを取得
                eeg_data = data[selected_channels, :]

                # タイムスタンプの計算
                num_samples = eeg_data.shape[1]
                timestamps = np.arange(num_samples) / self.board.get_sampling_rate(BoardIds.CYTON_BOARD)
                timestamps += time.time() - self.start_time

                # ラベルの取得
                labels = np.full(num_samples, self.current_label)

                eeg_data_list.append(eeg_data.T)
                timestamp_list.append(timestamps)
                label_list.append(labels)

            # データをまとめて処理
            eeg_data_concat = np.concatenate(eeg_data_list, axis=0)
            timestamp_concat = np.concatenate(timestamp_list)
            label_concat = np.concatenate(label_list)

            # データフレームの作成
            data = pd.DataFrame(eeg_data_concat, columns=[f'channel_{i+1}' for i in range(eeg_data_concat.shape[1])])
            data['timestamp'] = timestamp_concat
            data['label'] = label_concat

            # CSVファイルに保存
            data.to_csv(file_path, index=False)

        except BrainFlowError as e:
            logging.warning(e)

        finally:
            if self.board.is_prepared():
                self.board.stop_stream()
                self.board.release_session()
                print("End of EEG data collection")
            return
    
    def stop(self):
        self.stop_signal = True

def set_up_board():
    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument(
        "--timeout",
        type=int,
        help="timeout for device discovery or connection",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--ip-port", type=int, help="ip port", required=False, default=0
    )
    parser.add_argument(
        "--ip-protocol",
        type=int,
        help="ip protocol, check IpProtocolType enum",
        required=False,
        default=0,
    )
    parser.add_argument(
        "--ip-address", type=str, help="ip address", required=False, default=""
    )
    parser.add_argument(
        "--serial-port", type=str, help="serial port", required=True, default=""
    )
    parser.add_argument(
        "--mac-address", type=str, help="mac address", required=False, default=""
    )
    parser.add_argument(
        "--other-info", type=str, help="other info", required=False, default=""
    )
    parser.add_argument(
        "--streamer-params",
        type=str,
        help="streamer params",
        required=False,
        default="",
    )
    parser.add_argument(
        "--serial-number", type=str, help="serial number", required=False, default=""
    )
    parser.add_argument(
        "--board-id",
        type=int,
        help="board id, check docs to get a list of supported boards",
        required=False,
    )
    parser.add_argument("--file", type=str, help="file", required=False, default="")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    print("args: ", args)

    if args.log:
        BoardShim.enable_dev_board_logger()
    else:
        BoardShim.disable_board_logger()

    try:
        board = BoardShim(BoardIds.CYTON_BOARD, params)
        return board
    except BaseException as e:
        logging.warning("Exception", exc_info=True)


def main():
    board = set_up_board()

    app = QApplication(sys.argv)

    task = MotorImageryTask(board)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
