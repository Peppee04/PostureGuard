# PostureGuard

Real-time computer vision system for posture monitoring.

## Goal

PostureGuard uses a webcam and computer vision to monitor the user's sitting posture in real time.

The system analyzes body landmarks detected with MediaPipe and identifies excessive forward head movement. If an incorrect posture is maintained for a configurable amount of time, PostureGuard sends a Windows notification with an audio alert.

## Features

* Real-time webcam monitoring
* Human pose detection with MediaPipe
* Front posture analysis
* Personalized posture calibration
* Moving-average filtering to reduce false alarms
* Configurable bad-posture detection duration
* Windows desktop notifications with audio
* Background monitoring
* System tray integration
* Save and load custom calibrations
* OpenCV-based visualization

## Posture Analysis

PostureGuard uses MediaPipe Pose Landmarks to estimate the position of the:

* Nose
* Ears
* Shoulders
* Hips

For front-view analysis, the system evaluates the normalized distance between the head and shoulders.

A calibration procedure is used to personalize the detection threshold for each user.

## Calibration

At startup, PostureGuard can perform a personalized calibration consisting of:

1. Positioning
2. Normal posture
3. Slightly forward posture
4. Very forward posture

The measurements collected during calibration are used to calculate a personalized threshold.

Calibrations can be saved and loaded later, allowing the user to avoid repeating the calibration process.

## Monitoring

During monitoring, posture measurements are collected continuously.

A moving average is applied to the measurements to reduce the effect of short-term fluctuations and false detections.

If the user's posture remains below the calibrated threshold for longer than the configured duration, PostureGuard activates an alarm and sends a Windows notification.

## Project Structure

```text
PostureGuard/
├── src/
│   └── main.py
├── tests/
├── data/
│   └── .gitkeep
├── models/
│   ├── .gitkeep
│   └── pose_landmarker_full.task
├── calibrations/
│   └── .gitkeep
├── .gitignore
├── README.md
└── environment.yml
```

## Requirements

* Windows
* Python 3.11
* Webcam
* Conda (recommended)
* MediaPipe
* OpenCV
* NumPy
* Winotify
* Pystray
* Pillow

## Installation

Create the Conda environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate postureguard
```

Download the required MediaPipe pose model and place it in:

```text
models/pose_landmarker_full.task
```

## Running the Application

From the project root:

```bash
python src/main.py
```

PostureGuard will open the webcam and start the application.

## Controls

* `ENTER` — confirm a selection
* `ESC` — go back / exit the current screen
* `Q` — close PostureGuard completely

Closing the OpenCV window with the `X` button does not terminate the application. PostureGuard continues monitoring in the background and can be reopened through the system tray.

## Technologies

* Python
* OpenCV
* MediaPipe
* NumPy
* Winotify
* Pystray
* Pillow

## Project Status

**Version 1.0.0**

