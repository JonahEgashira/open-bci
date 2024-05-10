import pandas as pd
import numpy as np
import mne


def create_raw_from_csv(file_path, sfreq):
    data = pd.read_csv(file_path, header=None, sep="\t")
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
    raw = mne.io.RawArray(data, info)

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    return raw


def filter_raw(raw):
    raw.filter(l_freq=1, h_freq=100, verbose=True, fir_design="firwin")
    raw.notch_filter(freqs=np.arange(50, 101, 50), verbose=True, fir_design="firwin")
    return raw


def plot_raw(raw):
    plot = raw.plot(block=True, scalings="auto")

    return


def main():
    file_path = "./data/eeg_data.csv"
    sfreq = 250
    raw = create_raw_from_csv(file_path, sfreq)

    plot_raw(raw)

    return


if __name__ == "__main__":
    main()
