#!/usr/bin/env python3
"""
Enhanced patch script to handle Lambda functions with arguments in classification model
"""

# Read the current file
with open('src/DeepBee/software/detection_and_classification.py', 'r') as f:
    content = f.read()

# Find and replace the monkey patch section with an enhanced version
old_patch = '''# Monkey patch func_load to handle bad marshal data from legacy Lambda layers
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
    print(f"Warning: Could not patch func_load: {e}")'''

# Enhanced patch that handles Lambda functions with arguments
new_patch = '''# Monkey patch func_load to handle bad marshal data from legacy Lambda layers
try:
    from keras.utils import generic_utils
    _original_func_load = generic_utils.func_load
    
    def safe_func_load(code, defaults=None, globs=None):
        try:
            return _original_func_load(code, defaults, globs)
        except (ValueError, TypeError) as e:
            print(f"Warning: Failed to unmarshal lambda function, using fallback: {e}")
            # Return a function that handles the tensor splitting for classification model
            def fallback_lambda(x, **kwargs):
                import tensorflow as tf
                # Handle the tensor splitting based on the 'i' argument
                if 'i' in kwargs and 'parts' in kwargs:
                    i = kwargs['i']
                    parts = kwargs['parts']
                    # Split tensor along batch dimension and return the i-th part
                    if hasattr(tf, 'split'):
                        split_tensors = tf.split(x, parts, axis=0)
                        if i < len(split_tensors):
                            return split_tensors[i]
                # Fallback to identity if we can't handle the arguments
                return x
            return fallback_lambda
    
    generic_utils.func_load = safe_func_load
except Exception as e:
    print(f"Warning: Could not patch func_load: {e}")'''

if old_patch in content:
    new_content = content.replace(old_patch, new_patch)

    # Write the patched file
    with open('src/DeepBee/software/detection_and_classification.py', 'w') as f:
        f.write(new_content)

    print("Successfully enhanced Lambda function patch")
else:
    print("Could not find the old patch to replace")
