import numpy as np

def verify_coe_file(filepath, expected_shape):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return

    # 1. Verify the Vivado BRAM Header
    header_1 = lines[0].strip()
    header_2 = lines[1].strip()
    if "radix=16" not in header_1 or "vector=" not in header_2:
        print(f"FAILURE: {filepath} has an invalid Vivado header.")
        return
        
    # 2. Extract the raw hexadecimal data
    data_lines = lines[2:]
    raw_hex_string = "".join(data_lines).replace('\n', '').replace(';', '')
    hex_values = raw_hex_string.split(',')

    # 3. Convert Hex to Signed INT8 (handling Two's Complement)
    int8_weights = []
    for hx in hex_values:
        hx = hx.strip()
        if hx:
            val = int(hx, 16)
            if val > 127:  # Two's complement correction for negative numbers
                val -= 256
            int8_weights.append(val)
            
    weights_array = np.array(int8_weights, dtype=np.int8)
    
    print(f"--- Verification Report: {filepath} ---")
    print(f"Total Weights Extracted: {len(weights_array)}")
    
    # 4. Verify Hardware Memory Shape
    try:
        reshaped = weights_array.reshape(expected_shape)
        print(f"Shape Check: PASSED. Successfully mapped to {expected_shape}.")
        print(f"Data Range: Min = {np.min(reshaped)}, Max = {np.max(reshaped)} (Must be within -128 to +127)")
        print(f"Sample (First 5 weights): {weights_array[:5]}\n")
    except ValueError as e:
        print(f"SHAPE MISMATCH ERROR: Expected {expected_shape} but got {len(weights_array)} total weights.\n")

if __name__ == "__main__":
   
    
    # The expected shape for Conv1: 4 filters, 1 channel, 3x3 kernel = 36 total weights
    conv1_path = 'project/output files/conv1_weights.coe'
    verify_coe_file(conv1_path, (4, 1, 3, 3))

    # The expected shape for FC: 1 output node, 16,384 inputs = 16,384 total weights
    fc_path = 'project/output files/fc_weights.coe'
    verify_coe_file(fc_path, (1, 16384))