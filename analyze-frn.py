import os
import pandas as pd
import numpy as np
import mne

def create_raw_from_csv_pick(file_path, sfreq):
    data = pd.read_csv(file_path, header=None, sep="\t")

    print(f"Data shape before filtering: {data.shape}")

    picks_columns = [8]  # Corresponding to "Cz"

    timestamps = data.iloc[:, 0].values
    data = data.iloc[:, picks_columns]
    print(f"Data shape after filtering: {data.shape}")

    data = data.values.T

    ch_names = ["Cz"]
    ch_types = ["eeg"] * len(ch_names)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    raw.set_montage(
        mne.channels.make_standard_montage("standard_1020"), on_missing="ignore"
    )

    return raw, timestamps

def create_raw_from_csv(file_path, sfreq):
    data = pd.read_csv(file_path, header=None, sep="\t")

    print(f"Data shape: {data.shape}")

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


def create_epochs(raw, response_path, timestamps, sfreq):
    response_df = pd.read_csv(response_path, header=0)
    
    start_time = timestamps[0]
    events = []
    
    for _, row in response_df.iterrows():
        event_time = row["reaction_times"]
        sample_index = np.argmin(np.abs(timestamps - event_time)) # Find the closest timestamp sample
        events.append([sample_index, 0, int(row["correct_responses"])])
    
    events = np.array(events)
    
    event_id = dict(Correct=1, Incorrect=0)
    
    tmin = -2
    tmax = 4

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
    )
    
    return epochs
    

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
    response_path = os.path.join(latest_folder, "response_data.csv")
    sfreq = 250

    raw, timestamps = create_raw_from_csv_pick(file_path, sfreq)
    raw = filter_raw(raw)

    picks = ["Cz"]

    # plot_selected_channels(raw, picks=picks)

    epochs = create_epochs(raw, response_path, timestamps, sfreq)

    epochs.save("epoch.fif", overwrite=True)

    return

if __name__ == "__main__":
    main()
