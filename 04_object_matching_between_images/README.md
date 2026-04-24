# Object Matching Between Two Images

This project detects a common object between two images using the SIFT algorithm. The provided template image is matched against scene images, and the detected object instances are counted.

## Method
- Detect SIFT keypoints and descriptors in the template image
- Detect SIFT keypoints and descriptors in the scene image
- Match descriptors using BFMatcher
- Filter matches using Lowe's ratio test
- Estimate homography using RANSAC
- Localize and count detected object instances

## Project structure
- `data/` contains the template image and scene images
- `output/` contains saved results
- `src/` contains the implementation

## Run

Example for scene 1:
```bash
python -m src.main --template data/hp_wand.png --scene data/hp_scene1.jpg --output output/hp_scene1_detected.jpg --matches-output output/hp_scene1_matches.jpg --show