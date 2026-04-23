# Logo or Symbol Detection

This project detects high-contrast objects such as road signs or symbols in complex backgrounds using the MSER algorithm.

## Method
- Convert the image to grayscale
- Detect candidate regions using MSER
- Filter regions using size, shape, and fill constraints
- Merge nearby candidate regions
- Draw bounding boxes around the final detected sign regions

## Project structure
- `data/` contains the input images
- `output/` contains result images
- `src/` contains the implementation

## Run

Example:
```bash
python -m src.main --input data/road_signs.jpg --output output/road_signs_detected.jpg --show