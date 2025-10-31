#!/usr/bin/env python3
"""
Add debugging and error handling before prediction to catch issues earlier
"""

# Read the current file
with open('src/DeepBee/software/detection_and_classification.py', 'r') as f:
    content = f.read()

# Add debugging before prediction to catch issues
old_preprocess = '''        print("preprocess_input")
        blob_imgs = preprocess_input(blob_imgs)

        scores = None

        print("predict", len(blob_imgs))'''

new_preprocess = '''        print("preprocess_input")
        try:
            blob_imgs = preprocess_input(blob_imgs)
            print(f"Preprocess successful, blob_imgs shape: {blob_imgs.shape}, dtype: {blob_imgs.dtype}")
        except Exception as preprocess_error:
            print(f"Preprocess failed: {preprocess_error}")
            raise

        scores = None

        print(f"Starting prediction with {len(blob_imgs)} images, batch_size: {batch_size}")
        
        # Check if we have any images to process
        if len(blob_imgs) == 0:
            print("No images to classify, returning empty results")
            return np.array([])'''

if old_preprocess in content:
    new_content = content.replace(old_preprocess, new_preprocess)

    # Write the patched file
    with open('src/DeepBee/software/detection_and_classification.py', 'w') as f:
        f.write(new_content)

    print("Successfully added debugging before prediction")
else:
    print("Could not find preprocess code to patch")
    print("Current content around preprocess_input:")
    # Show current content around preprocess_input
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'preprocess_input' in line:
            start = max(0, i-3)
            end = min(len(lines), i+8)
            for j in range(start, end):
                prefix = ">>>" if j == i else "   "
                print(f"{prefix} {j}: {lines[j]}")
            break
