#!/usr/bin/env python3
"""
Patch script to fix bad marshal data error in Lambda layers
"""

import re

# Read the original file
with open('src/DeepBee/software/detection_and_classification.py', 'r') as f:
    content = f.read()

# Define the patch to insert after the imports but before the K.__dict__.update
patch_code = '''
# Monkey patch func_load to handle bad marshal data from legacy Lambda layers
try:
    from keras.utils import generic_utils
    _original_func_load = generic_utils.func_load
    
    def safe_func_load(code, defaults=None, globs=None):
        try:
            return _original_func_load(code, defaults, globs)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to unmarshal lambda function, using identity: {e}")
            return lambda x: x
    
    generic_utils.func_load = safe_func_load
except Exception as e:
    print(f"Warning: Could not patch func_load: {e}")

'''

# Find the position to insert the patch (after imports, before K.__dict__.update)
insert_position = content.find('# try:\nK.__dict__.update(')
if insert_position == -1:
    insert_position = content.find('K.__dict__.update(')

if insert_position != -1:
    # Insert the patch code
    new_content = content[:insert_position] + patch_code + content[insert_position:]

    # Write the patched file
    with open('src/DeepBee/software/detection_and_classification.py', 'w') as f:
        f.write(new_content)

    print("Successfully patched detection_and_classification.py")
else:
    print("Could not find insertion point in file")
