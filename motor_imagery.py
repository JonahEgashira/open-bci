import argparse
import time
import sys
import numpy as np
import logging
import os

from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds,
    BrainFlowError,
)

from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox, QProgressBar
from PyQt5.QtCore import Qt, QEvent, QTimer


DEFAULT_CHANNELS = [7]  # C3 channel


class MotorImageryTask(QWidget):
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.directory_handler = DirectoryHandler()
        self.eeg_handler = EEGHandler(
            self.board, self.directory_handler, DEFAULT_CHANNELS
        )
        self.eeg_handler.setup_and_prepare_session()
        self.eeg_handler.start_stream()

        self.trials = 2
        self.trial_count = 0
        self.width = None
        self.height = None
        self.waiting_for_start = True
        self.start_wait_time = 2000
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
        self.eeg_handler.clear_buffer()
        self.eeg_handler.set_start_time()
        self.showInstruction()
        return

    def keyPressEvent(self, event: QEvent) -> None:
        if self.waiting_for_start:
            self.waiting_for_start = False
            self.directory_handler.create_directory()
            self.eeg_handler.create_data_file()
            self.instruction_label.setText("Starting...")
            self.instruction_label.show()
            QTimer.singleShot(self.start_wait_time, self.startTask)
            return

    def showInstruction(self):
        print(f"Trial count: {self.trial_count}")
        current_time = time.time()
        current_time_formatted = datetime.fromtimestamp(current_time).strftime("%H%M%S")
        print(f"Current time: {current_time_formatted}")
        instruction_number = 4

        if self.trial_count >= self.trials * instruction_number:
            self.end_task()
            return

        if self.trial_count % instruction_number == 0:
            instruction = "Imagine"
            label = 0
            wait_time = 5000
        elif self.trial_count % instruction_number == 1:
            instruction = "Relax"
            label = 1
            wait_time = 2000
        elif self.trial_count % instruction_number == 2:
            instruction = "Rest"
            label = 2
            wait_time = 5000
        else:
            instruction = "Ready..."
            label = 3
            wait_time = 2000

        self.instruction_label.setText(instruction)
        self.instruction_label.show()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(
            (self.width - self.width // 2) // 2,
            self.height // 2 + 50,
            self.width // 2,
            30,
        )
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(wait_time)
        self.progress_bar.setValue(wait_time)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.show()

        self.trial_count += 1

        print(f"Wait time: {wait_time}")

        self.eeg_handler.start_data_collection(label)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress_bar)
        self.timer.start(10)

    def update_progress_bar(self):
        current_value = self.progress_bar.value()
        if current_value > 0:
            self.progress_bar.setValue(current_value - 10)
        else:
            self.timer.stop()
            self.progress_bar.hide()
            self.stop_data_collection()

    def stop_data_collection(self):
        self.eeg_handler.stop_data_collection()
        self.showInstruction()

    def end_task(self):
        QMessageBox.information(self, "Info", "Fin.")
        self.eeg_handler.stop()
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
        self.sfreq = self.board.get_sampling_rate(BoardIds.CYTON_BOARD)
        self.directory_handler = directory_handler
        self.start_time = None
        self.data_storage = []
        self.file_path = None
        if channels is None:
            self.channels = [1, 2, 3, 4, 5, 6, 7, 8]
        else:
            self.channels = channels

    def setup_and_prepare_session(self):
        try:
            self.board.prepare_session()
            print("Session prepared")
        except BrainFlowError as e:
            logging.warning(e)

    def start_stream(self):
        try:
            self.board.start_stream()
            print("Start streaming")
            print("Sfreq: ", self.board.get_sampling_rate(BoardIds.CYTON_BOARD))
        except BrainFlowError as e:
            logging.warning(e)

    def clear_buffer(self):
        self.board.get_board_data()

    def set_start_time(self):
        self.start_time = time.time()
        start_time_formatted = datetime.fromtimestamp(self.start_time).strftime(
            "%H%M%S"
        )
        print(f"Start time: {start_time_formatted}")

    def create_data_file(self):
        data_dir = "./data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        directory_path = self.directory_handler.get_directory_path()
        file_path = f"{directory_path}/eeg_data.csv"

        self.file_path = file_path

    def get_data_file(self):
        return self.file_path

    def start_data_collection(self, label):
        self.current_label = label
        self.data_collection_timer = QTimer()
        self.data_collection_timer.timeout.connect(self.collect_data)
        self.data_collection_timer.start(50)

    def collect_data(self):
        data = self.board.get_board_data()

        eeg_channels = self.board.get_eeg_channels(self.board.board_id)
        selected_channels = [eeg_channels[c - 1] for c in self.channels]
        eeg_data = data[selected_channels, :]

        num_samples = eeg_data.shape[1]
        timestamps = np.arange(num_samples) / self.board.get_sampling_rate(
            BoardIds.CYTON_BOARD
        )
        timestamps += time.time() - self.start_time

        labels = np.full(num_samples, self.current_label)
        eeg_data_with_labels = np.column_stack((eeg_data.T, timestamps, labels))

        self.data_storage.append(eeg_data_with_labels)

    def stop_data_collection(self):
        self.data_collection_timer.stop()

    def save_data(self):
        file_path = self.get_data_file()
        data = np.concatenate(self.data_storage, axis=0)

        # ラベルの列を整数に変換
        data[:, -1] = data[:, -1].astype(int)

        # fmt で各列のフォーマットを指定
        np.savetxt(file_path, data, delimiter=",", fmt=["%.3f", "%.3f", "%d"])
        print("Data saved to eeg_data.csv")

    def stop(self):
        self.save_data()

        if self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            print("End of EEG data collection")

        return


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
