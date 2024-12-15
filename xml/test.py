import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Test OCR
from PIL import Image
image = Image.open('sentences/a01/a01-000u/a01-000u-s00-02.png')
data = pytesseract.image_to_boxes(image, lang='eng')

# Parse bounding boxes
boxes = []
for box in data.splitlines():
    char, x, y, w, h, _ = box.split()
    boxes.append((int(x), int(y), int(w), int(h)))

# Group boxes into desired number of segments
desired_segments = 29
group_size = max(1, len(boxes) // desired_segments)
segments = [boxes[i:i + group_size] for i in range(0, len(boxes), group_size)]

# Crop and save each segment
for i, group in enumerate(segments):
    if len(group) == 0:
        continue  # Skip empty groups
    
    # Calculate bounding box for the group
    x_min = min(b[0] for b in group)
    y_min = min(b[1] for b in group)
    x_max = max(b[2] for b in group)
    y_max = max(b[3] for b in group)

    # Adjust coordinates for PIL (invert Y-axis)
    top = image.height - y_max
    bottom = image.height - y_min
    segment = image.crop((x_min, top, x_max, bottom))
    segment.save(f'segment_{i}.png')




