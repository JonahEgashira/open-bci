
# https://brainflow.readthedocs.io/en/stable/Examples.html#python
# 杉野さん、出利葉さんからもらったファイル

import time
import logging
import random


import PyQt6
import sys
from PyQt6.QtWidgets import QApplication, QWidget

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, DetrendOperations #WindowFunctions, 
from pythonosc import udp_client, osc_message_builder

openframeworks_ip = "127.0.0.1"
openframeworks_port = 9000
#client = udp_client.SimpleUDPClient(openframeworks_ip, openframeworks_port)
#message = osc_message_builder.OscMessageBuilder(address="/data")
#ch1 = osc_message_builder.OscMessageBuilder(address="/data/ch1")

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        channels_use = BoardShim.get_exg_channels(self.board_id)
        self.exg_channels = channels_use[0:len(channels_use)-1]
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50 #timerの更新時間感覚（ms）
        self.window_size = 4
        self.num_points = self.window_size * self.sampling_rate

        print(self.sampling_rate)

        self.app = QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True,title='BrainFlow Plot',size=(2000, 1000))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)#タイマーによってupdateを定期的に呼び出す
        timer.start(self.update_speed_ms)

        QApplication.instance().exec()


    def _init_timeseries(self): #ボード、グラフの初期化、タイマーの設定
        self.plots = list()
        self.curves = list()
        for i in range(len(self.exg_channels)):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            p.setYRange(-100,100,padding=0) #y軸のレンジ
            p.setXRange(self.sampling_rate,self.num_points+self.sampling_rate,padding=0)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        data = self.board_shim.get_current_board_data(self.num_points+self.sampling_rate)#num_points分データを取得？samplingrate分足してるのは？
        # datanow = data[-4000:-1,self.exg_channels]
        avg_bands = [0, 0, 0, 0, 0]
        #filter
        for count, channel in enumerate(self.exg_channels):
            # plot timeseries
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], self.sampling_rate, 2.0, 49.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            #DataFilter.perform_bandpass(data[channel], self.sampling_rate, 26.0, 50.0, 2,
            #                            FilterTypes.BUTTERWORTH.value, 0)
            #4-50Hzをバンドストップにしているのはなぜ？
            DataFilter.perform_bandstop(data[channel], self.sampling_rate, 48.0, 52.0, 2,
                                        FilterTypes.BUTTERWORTH.value, 0)
            #DataFilter.perform_bandstop(data[channel], self.sampling_rate, 4.0, 50.0, 2,
            #                            FilterTypes.BUTTERWORTH.value, 0)
            self.curves[count].setData(data[channel].tolist()) #カーブオブジェクトでの波形の表示

            #if count == 1:
            #    send_osc_message("/data/ch1", data[channel].tolist())

            #ch1.add_arg(data[channel].tolist())
            #ch1 = ch1.build()

        self.app.processEvents()#eventを処理してグラフが更新されるように
        
        # OSCメッセージの作成
        #message.add_arg(data.tolist())  # データを追加（ここでは例として42を使用）
        #message.add_arg(42.0) 
        #print(data.tolist())

        # メッセージを送信
        #client.send_message(message)

def send_osc_message(address, data):
    # OSC通信先の設定
    openframeworks_ip = "127.0.0.1"  # OpenFrameworksのIPアドレス
    openframeworks_port = 9000  # OpenFrameworksのポート番号

    # OSCクライアントの作成
    client = udp_client.SimpleUDPClient(openframeworks_ip, openframeworks_port)

    # OSCメッセージの送信
    client.send_message(address, data)


def main():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbserial-4'
    #params.serial_port = '/dev/cu.usbserial-DM03GV5S'
    
    ####
    subname = "saitou1" #保存の名前を変える
    ####

    try:
        board_shim = BoardShim(BoardIds.CYTON_BOARD, params)
        #board_shim = BoardShim(0, params)
        board_shim.prepare_session()
        board_shim.start_stream() #ストリームの開始
        g = Graph(board_shim) #グラフクラスへボードストリームをセットする
    except BaseException as e:
        logging.warning('Exception', exc_info=True)#エラーが出た際のログの表示
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            row_data = board_shim.get_board_data() #save data
            DataFilter.write_file(row_data, subname+".csv", 'w')  # use 'a' for append mode
            board_shim.stop_stream()
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    main()
    
    
