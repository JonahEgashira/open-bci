import argparse
import time
import numpy as np
import logging

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
    WindowOperations,
)

import pyqtgraph as pg
import sys

from PyQt5.QtWidgets import QApplication, QWidget
from pyqtgraph.Qt import QtCore, QtGui
from pythonosc import udp_client, osc_message_builder

import mne


class Graph:
    def __init__(self, board_shim):
        self.board_shim = board_shim
        self.board_id = board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)[:-1]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 4  # seconds
        self.num_points = self.window_size * self.sampling_rate

        print(f"Sampling Rate: {self.sampling_rate}")

        self.app = QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(
            show=True, title="BrainFlow Plot", size=(800, 600)
        )
        self._init_psd_plots()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.app.exec_()

    def _init_psd_plots(self):
        self.psd_plot = self.win.addPlot(title="All Channels FFT")
        self.psd_plot.showGrid(x=True, y=True)
        self.psd_plot.setXRange(1, 60)
        self.psd_plot.setLogMode(x=False, y=True)  # Y軸のみ対数スケールに設定
        self.psd_plot.setYRange(np.log10(0.1), np.log10(100))
        self.psd_curves = []
        colors = ["r", "g", "b", "c", "m", "y", "w"]  # 色のリスト

        for i, channel in enumerate(self.exg_channels):
            curve = self.psd_plot.plot(pen=colors[i % len(colors)])  # 色を循環利用
            self.psd_curves.append(curve)

    def apply_smoothing(self, data, alpha=0.9):
        smoothed_data = np.zeros_like(data)
        smoothed_data[0] = data[0]
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
        return smoothed_data

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points)
        for count, channel in enumerate(self.exg_channels):
            channel_data = data[
                channel, -self.num_points :
            ]  # 最新のデータポイントを取得
            if len(channel_data) % 2 != 0:
                channel_data = channel_data[:-1]  # データの長さを偶数に調整

            DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(
                channel_data,
                self.sampling_rate,
                2.0,
                49.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )
            DataFilter.perform_bandstop(
                channel_data,
                self.sampling_rate,
                48.0,
                52.0,
                2,
                FilterTypes.BUTTERWORTH.value,
                0,
            )

            # FFTを計算
            fft_data = DataFilter.perform_fft(channel_data, window=WindowOperations.HANNING)
            # 周波数軸のデータを生成
            freqs = np.linspace(0, self.sampling_rate / 2, len(fft_data))

            # FFTデータの振幅を計算し、平滑化を適用
            fft_amplitudes = np.abs(fft_data)
            smoothed_fft_amplitudes = self.apply_smoothing(fft_amplitudes, alpha=0.9)

            # FFTデータの振幅のみを取得し、プロット
            self.psd_curves[count].setData(
                freqs, smoothed_fft_amplitudes
            )  # FFT振幅を更新

        self.app.processEvents()  # グラフを更新


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


def get_eeg_data(board):
    eeg_data = None

    try:
        board.prepare_session()
        board.start_stream()

        graph = Graph(board_shim=board)

        data = board.get_board_data()
        eeg_channels = board.get_eeg_channels(board.board_id)

        eeg_data = data[eeg_channels, :]

        DataFilter.write_file(eeg_data, "eeg_data.csv", "w")

        print("Data saved to eeg_data.csv")

    except BrainFlowError as e:
        logging.warning("Exception", exc_info=True)

    finally:
        logging.info("End")
        if board.is_prepared():
            board.stop_stream()
            logging.info("Releasing session")
            board.release_session()

        logging.info("Session released")
        return eeg_data


def main():
    board = set_up_board()
    eeg_data = get_eeg_data(board)


if __name__ == "__main__":
    main()
