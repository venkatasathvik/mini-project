import cv2
import numpy as np
import os
import glob

# Configuration
TILE_SIZE = 128

# Mapped to your new folder structure
INPUT_DIR = 'initial dataset/skypulse dataset' 
OUTPUT_DIR = 'test_dataset_of_skypulse/clear' 

# Ensure directories exist
os.makedirs('test_dataset_of_skypulse/clear', exist_ok=True)
os.makedirs('test_dataset_of_skypulse/cloudy', exist_ok=True)

def slice_skypulse_telemetry():
    tile_count = 0
    
    image_files = glob.glob(os.path.join(INPUT_DIR, '*.jpg')) + glob.glob(os.path.join(INPUT_DIR, '*.png'))
    
    if not image_files:
        print(f"No images found in '{INPUT_DIR}'.")
        return
        
    for img_path in image_files:
        print(f"Slicing {img_path}...")
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        h, w = img.shape
        
        for y in range(0, h - TILE_SIZE + 1, TILE_SIZE):
            for x in range(0, w - TILE_SIZE + 1, TILE_SIZE):
                
                patch = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                scaled_patch = np.round((patch / 255.0) * 127.0).astype(np.uint8)
                
                save_path = os.path.join(OUTPUT_DIR, f"skypulse_{tile_count:04d}.png")
                cv2.imwrite(save_path, scaled_patch)
                tile_count += 1
                
    print(f"\nSuccessfully extracted {tile_count} INT8-scaled tiles into 'test_dataset_of_skypulse/clear'.")
    print("ACTION REQUIRED: Manually review these tiles and move the cloudy ones into 'test_dataset_of_skypulse/cloudy'.")

if __name__ == "__main__":
    slice_skypulse_telemetry()