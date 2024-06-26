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
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        print(f"Sampling Rate: {self.sampling_rate}")

        self.app = QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(
            show=True, title="BrainFlow Plot", size=(2000, 1000)
        )
        self._init_timeseries()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(self.update_speed_ms)

        self.app.exec_()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i, channel in enumerate(self.exg_channels):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis("left", False)
            p.setMenuEnabled(False)
            p.setYRange(-100, 100, padding=0)
            p.setXRange(
                self.sampling_rate, self.num_points + self.sampling_rate, padding=0
            )
            if i == 0:
                p.setTitle("TimeSeries Plot")
            curve = p.plot()
            self.plots.append(p)
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(
            self.num_points + self.sampling_rate
        )
        for count, channel in enumerate(self.exg_channels):
            channel_data = data[channel]
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
            self.curves[count].setData(channel_data.tolist())
        self.app.processEvents()  # Update the graph


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


def analyze_eeg_data(board, eeg_data):
    eeg_data = eeg_data / 1e6  # BrainFlow returns uV, convert to V for MNE

    ch_names = [
        'N/A', # First channel is not connected
        'O2',
        'O1',
        'Pz',
        'C4',
        'Cz',
        'C3',
        'Fz',
    ]
    ch_types = ['eeg'] * len(ch_names)

    sfreq = board.get_sampling_rate(board.board_id)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)

    plot = raw.plot(
        block=True,
        scalings='auto'
    )

def get_eeg_data(board):
    eeg_data = None

    try:
        board.prepare_session()
        board.start_stream()

        graph = Graph(board_shim=board)

        data = board.get_board_data()
        eeg_channels  = board.get_eeg_channels(board.board_id)

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
    analyze_eeg_data(board, eeg_data)
    
if __name__ == "__main__":
    main()
