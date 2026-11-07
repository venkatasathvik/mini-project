import numpy as np
import cv2
import os

def load_weights_from_coe(filepath, shape):
    """Reads Vivado Hex .coe file and converts back to an INT8 NumPy array."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    data_lines = lines[2:] # Skip the radix and vector headers
    raw_hex = "".join(data_lines).replace('\n', '').replace(';', '')
    hex_list = raw_hex.split(',')
    
    int8_list = []
    for hx in hex_list:
        hx = hx.strip()
        if hx:
            val = int(hx, 16)
            if val > 127: # Two's complement conversion
                val -= 256
            int8_list.append(val)
            
    return np.array(int8_list, dtype=np.int8).reshape(shape)

def simulate_hardware_forward_pass(image_path, conv_weights, fc_weights):
    """Performs the exact INT8 math your Verilog will execute."""
    
    print(f"\n--- Testing Image: {os.path.basename(image_path)} ---")
    
    # 1. Load Image as Grayscale (0-255)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return
        
    # 2. Hardware Preprocessing: Scale to 0-127
    img_int8 = np.round((img / 255.0) * 127.0).astype(np.int8)
    
    # 3. Convolution Layer (3x3, Padding=1, Stride=1)
    # Using np.pad to simulate hardware boundary handling (zero padding)
    img_padded = np.pad(img_int8, pad_width=1, mode='constant', constant_values=0)
    
    # The output feature map for 4 filters on a 128x128 image
    conv_out = np.zeros((4, 128, 128), dtype=np.int32) 
    
    print("Executing INT8 Convolution...")
    for f in range(4): # 4 Filters
        for y in range(128):
            for x in range(128):
                # Extract the 3x3 window from the padded image
                window = img_padded[y:y+3, x:x+3]
                # Extract the 3x3 weight filter
                filter_w = conv_weights[f, 0, :, :]
                
                # Perform the Multiply-Accumulate (MAC)
                mac_result = np.sum(window.astype(np.int32) * filter_w.astype(np.int32))
                conv_out[f, y, x] = mac_result
                
    # 4. ReLU Activation (Hardware sets negative values to 0)
    relu_out = np.maximum(conv_out, 0)
    
    # 5. Max Pooling (2x2, Stride=2)
    pool_out = np.zeros((4, 64, 64), dtype=np.int32)
    for f in range(4):
        for y in range(0, 128, 2):
            for x in range(0, 128, 2):
                pool_out[f, y//2, x//2] = np.max(relu_out[f, y:y+2, x:x+2])
                
    # 6. Fully Connected Layer (Flatten and MAC)
    flat_out = pool_out.flatten()
    print("Executing INT8 Fully Connected MAC...")
    
    # Perform the final MAC for the single output node
    final_logit = np.sum(flat_out.astype(np.int64) * fc_weights.flatten().astype(np.int64))
    
    # 7. Hardware Decision Logic
    print(f"Final Hardware Accumulation (Logit): {final_logit}")
    if final_logit > 0:
        print("Hardware Classification: CLOUDY (1)")
    else:
        print("Hardware Classification: CLEAR (0)")

if __name__ == "__main__":
    # Load the weights directly from the .coe files
    print("Loading INT8 weights from .coe files...")
    conv1_weights = load_weights_from_coe('project/output files/conv1_weights.coe', (4, 1, 3, 3))
    fc_weights = load_weights_from_coe('project/output files/fc_weights.coe', (1, 16384))
    
    # --- TEST IMAGES ---
    # Pick one clear image and one cloudy image from your dataset to test
    # Replace these paths with actual image names in your folders
    clear_image = 'dataset/clear/tile_0000.png' 
    cloudy_image = 'dataset/cloudy/tile_0000.png'
    
    simulate_hardware_forward_pass(clear_image, conv1_weights, fc_weights)
    simulate_hardware_forward_pass(cloudy_image, conv1_weights, fc_weights)