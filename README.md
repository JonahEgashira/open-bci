## 使い方 (Mac)

- Python3を好きな方法でインストール
    - 既にインストールされているかの確認
    - `python -V` または `python3 -V` とターミナルに入力して `Python 3.x.x` の表示が出れば、既にインストールされています

    - こだわりがなければ公式サイトからのインストールが楽
    - https://www.python.org/downloads/macos/
    - 動作検証したPythonのバージョンは3.11.9ですが最新版でOK

- 仮想環境を作成（スキップ可）
    `python -m venv venv`
    `source ./venv/bin/activate`

- ライブラリのインストール

    `pip install -r requirements.txt`

- スクリプトを実行可能にする
    `chmod+x run_task_record.sh`

- 実行  `./run_task_record`
    - run_task_record.sh の中でOpenBCIドングルを認識して、task_record.py が呼ばれています。