# 04 — Object Matching Between Images

Detects and counts **wands** in Harry Potter scene images using the **SIFT** (Scale-Invariant Feature Transform) algorithm.

---

## Project Structure

```
04_object_matching_between_images/
├── data/
│   ├── hp_wand.png        ← reference object (query image)
│   ├── hp_scene1.jpg
│   ├── hp_scene2.jpg
│   ├── hp_scene3.jpg
│   └── hp_scene4.jpg
├── output/
│   └── wand_detection_result.png   ← generated after running
├── src/
│   ├── __init__.py
│   ├── main.py            ← entry point
│   ├── matcher.py         ← SIFT detection & matching logic
│   └── utils.py           ← image I/O & visualization helpers
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Run with defaults (reads from data/, writes to output/)
python src/main.py

# Custom options
python src/main.py --ratio 0.75       # stricter Lowe ratio
python src/main.py --min-match 4      # require more matches per detection
python src/main.py --data my_images/  # different image folder
```

---

## How It Works

### 1. Keypoint Detection
SIFT scans both the reference wand image and each scene image, finding
distinctive local features (corners, blobs) that are stable under changes
in **scale**, **rotation**, and **lighting**.

### 2. Descriptor Computation
Each keypoint gets a **128-dimensional descriptor** vector that encodes the
local gradient structure around it. This descriptor is what makes SIFT
scale- and rotation-invariant.

### 3. FLANN Matching
A **FLANN** (Fast Library for Approximate Nearest Neighbors) matcher finds
the closest descriptor pairs between the wand template and the scene.

### 4. Lowe's Ratio Test
For each match, the ratio of the best match distance to the second-best is
checked. If `best / second < ratio` (default 0.78), the match is accepted as
**good**. This removes ambiguous matches.

### 5. Spatial Clustering
Matched keypoint locations in the scene are grouped by proximity. Each
spatial cluster = **one detected wand instance**.

---

## Results

| Scene | Keypoints | Good Matches | Wands Detected |
|-------|-----------|--------------|----------------|
| hp_scene1 | 34,281 | 6 | 3 |
| hp_scene2 | 12,614 | 2 | 0 |
| hp_scene3 | 15,019 | 2 | 0 |
| hp_scene4 | 1,087  | 1 | 0 |
| **Total** | | | **3** |

---

## Limitations

SIFT works well for **instance matching** (finding the *same* object under
different viewpoints). It struggles with **class-level detection** (finding
*any* wand) because in-scene wands differ from the reference due to motion
blur, occlusion, lighting, and glowing spell effects. Deep learning approaches
(e.g. YOLO, Faster-RCNN) handle class-level detection better.
