#!/usr/bin/env python3
"""
Patch script to fix OpenCV 4.x compatibility for findContours
"""

# Read the current file
with open('src/DeepBee/software/detection_and_classification.py', 'r') as f:
    content = f.read()

# Replace the problematic findContours line
old_line = '    _, contours, _ = cv2.findContours(reconstructed_mask, 1, 2)'
new_lines = '''    # Handle both OpenCV 3.x (3 values) and 4.x (2 values) compatibility
    contours_result = cv2.findContours(reconstructed_mask, 1, 2)
    contours = contours_result[-2]  # contours is always second-to-last'''

if old_line in content:
    new_content = content.replace(old_line, new_lines)

    # Write the patched file
    with open('src/DeepBee/software/detection_and_classification.py', 'w') as f:
        f.write(new_content)

    print("Successfully patched OpenCV findContours compatibility")
else:
    print("Could not find the findContours line to patch")
