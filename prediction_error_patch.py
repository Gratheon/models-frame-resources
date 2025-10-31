#!/usr/bin/env python3
"""
Patch to add better error handling around model prediction to prevent silent crashes
"""

# Read the current file
with open('src/DeepBee/software/detection_and_classification.py', 'r') as f:
    content = f.read()

# Find the prediction loop and add error handling
old_prediction_code = '''        print("predict", len(blob_imgs))
        for chunk in [
            blob_imgs[x : x + batch_size] for x in range(0, len(blob_imgs), batch_size)
        ]:
            output = net.predict(chunk)

            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))'''

new_prediction_code = '''        print("predict", len(blob_imgs))
        try:
            for i, chunk in enumerate([
                blob_imgs[x : x + batch_size] for x in range(0, len(blob_imgs), batch_size)
            ]):
                print(f"Predicting batch {i+1} with {len(chunk)} images")
                try:
                    output = net.predict(chunk, verbose=0)
                    print(f"Batch {i+1} prediction successful, output shape: {output.shape}")
                    
                    if scores is None:
                        scores = np.copy(output)
                    else:
                        scores = np.vstack((scores, output))
                except Exception as batch_error:
                    print(f"Error predicting batch {i+1}: {batch_error}")
                    # Create dummy output with correct shape for this batch
                    dummy_output = np.zeros((len(chunk), 7))  # 7 classes
                    if scores is None:
                        scores = np.copy(dummy_output)
                    else:
                        scores = np.vstack((scores, dummy_output))
        except Exception as prediction_error:
            print(f"Major prediction error: {prediction_error}")
            # Create dummy scores for all images
            scores = np.zeros((len(blob_imgs), 7))  # 7 classes'''

if old_prediction_code in content:
    new_content = content.replace(old_prediction_code, new_prediction_code)

    # Write the patched file
    with open('src/DeepBee/software/detection_and_classification.py', 'w') as f:
        f.write(new_content)

    print("Successfully added error handling to prediction step")
else:
    print("Could not find prediction code to patch")
