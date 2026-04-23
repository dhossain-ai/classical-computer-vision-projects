# Text Detection in an Image

This project detects printed text regions in an image using the MSER algorithm and marks the detected text blocks.

## Method
- Convert the image to grayscale
- Detect candidate text regions using MSER
- Filter noisy regions using geometric constraints
- Merge nearby candidate regions into text blocks
- Draw bounding boxes around the final text regions

## Project structure
- `data/` contains the input image
- `output/` contains result images
- `src/` contains the implementation

## Run

```bash
python -m src.main --input data/text.jpg --output output/text_detected.jpg --show