#!/bin/bash

# Search for the first device that includes 'usbserial' in /dev/cu.*
device=$(ls /dev/cu.* | grep 'usbserial' | head -n 1)

# Check if a device was found
if [ -z "$device" ]; then
    echo "No usbserial device found."
    exit 1
else
    echo "Found device: $device"
    # Execute the Python script with the device ID as an argument
    python3 record_eeg.py --serial-port "$device"
fi
