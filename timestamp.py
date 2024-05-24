from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes
import time
import numpy as np


def main():
    # ボードIDとパラメータの設定
    board_id = BoardIds.CYTON_BOARD
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-D200PPR9"

    # ボードの初期化
    board_shim = BoardShim(board_id, params)
    board_shim.prepare_session()

    # ストリーミングの開始
    board_shim.start_stream()
    time.sleep(10)  # 10秒間データを収集
    board_shim.stop_stream()

    # データの取得
    data = board_shim.get_board_data()
    board_shim.release_session()

    # タイムスタンプを記録している列番号の取得
    timestamp_channel = BoardShim.get_timestamp_channel(board_id)

    print("Timestamp channel: ", timestamp_channel)

    eeg_channels = BoardShim.get_eeg_channels(board_id)
    print("EEG channels: ", eeg_channels)

    # データの分離
    timestamps = data[timestamp_channel]
    eeg_data = data[eeg_channels]

    # データの整形
    eeg_data_with_timestamps = np.concatenate(
        (timestamps.reshape(1, -1), frequencies.reshape(1, -1), eeg_data), axis=0
    )

    # データの保存
    DataFilter.write_file(eeg_data_with_timestamps, "brainwave_data.csv", "w")


if __name__ == "__main__":
    main()
