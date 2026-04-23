# Moving Object Tracking in Video

This project tracks a moving vehicle in a video using the Harris-Stephens corner detector and Lucas-Kanade optical flow. It estimates approximate speed and displays it on the video.

## Method
- Harris-Stephens corner detection
- Lucas-Kanade optical flow tracking
- Approximate motion speed estimation
- Speed overlay on output video

## Project structure
- `data/` contains input videos
- `output/` contains processed results
- `src/` contains the implementation

## How to run

For the first video:
```bash
python -m src.main --input data/moving_car.mp4 --output output/moving_car_result.mp4