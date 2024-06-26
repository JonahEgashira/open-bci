import os
import pandas as pd
import numpy as np
import mne

def create_raw_from_csv_pick(file_path, sfreq):
    data = pd.read_csv(file_path, header=None, sep="\t")

    print(f"Data shape before filtering: {data.shape}")

    picks_columns = [8]  # Corresponding to "Cz"
    data = data.iloc[:, picks_columns]
    print(f"Data shape after filtering: {data.shape}")

    data = data / 1e6
    data = data.values.T

    ch_names = ["Cz"]
    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    return raw

def create_raw_from_csv(file_path, sfreq):
    data = pd.read_csv(file_path, header=None, sep="\t")

    print(f"Data shape: {data.shape}")

    data = data / 1e6  # BrainFlow returns data in microvolts, convert to volts for MNE
    data = data.values.T

    ch_names = [
        "N/A",  # First channel is not connected
        "O2",
        "O1",
        "Pz",
        "C4",
        "Cz",
        "C3",
        "Fz",
    ]
    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=True)

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    return raw


def filter_raw(raw):
    raw.filter(l_freq=2.0, h_freq=49.0, picks="eeg", method="iir", verbose=True)
    raw.notch_filter(
        freqs=np.arange(50, 51, 50), picks="eeg", method="iir", verbose=True
    )

    return raw


def plot_selected_channels(raw, picks=None):
    if picks:
        raw.plot(block=True, scalings="auto", picks=picks)
    else:
        raw.plot(block=True, scalings="auto")

    return


def main():
    data_dir = "./data"
    folders = [os.path.join(data_dir, folder) for folder in os.listdir(data_dir)]
    folders = [folder for folder in folders if os.path.isdir(folder)]
    latest_folder = max(folders, key=os.path.getmtime)

    print(f"Latest folder: {latest_folder}")

    file_path = os.path.join(latest_folder, "eeg_data.csv")
    sfreq = 250

    raw = create_raw_from_csv_pick(file_path, sfreq)
    raw = filter_raw(raw)

    picks = ["Cz"]

    plot_selected_channels(raw, picks=picks)

    return


if __name__ == "__main__":
    main()
