import pandas as pd
import numpy as np
from typing import List, Dict

def create_epochs(data) -> List[Dict]:
    epochs = []
    current_epoch = []
    current_label = None

    for _, row in data.iterrows():
        label = row["label"]

        if current_label is None:
            current_label = label

        if label != current_label:
            epoch = {
                "eeg_data": np.array(current_epoch),
                "label": current_label,
            }
            epochs.append(epoch)
            current_epoch = []
            current_label = label

        current_epoch.append(row["channel_1"])


    if len(current_epoch) > 0:
        epoch = {
            "eeg_data": np.array(current_epoch),
            "label": current_label,
        }
        epochs.append(epoch)

    return epochs


data = pd.read_csv("eeg_data.csv")

epochs = create_epochs(data)

for epoch in epochs:
    print(epoch["eeg_data"].shape, epoch["label"])

