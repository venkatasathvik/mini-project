import cv2
import numpy as np
import os
import glob

# Configuration
TILE_SIZE = 128
TARGET_SAMPLES = 1000 # 1000 clear, 1000 cloudy

# Setup directories mapped exactly to your updated VS Code structure
OUTPUT_DIR_CLEAR = 'dataset/clear'
OUTPUT_DIR_CLOUDY = 'dataset/cloudy'
os.makedirs(OUTPUT_DIR_CLEAR, exist_ok=True)
os.makedirs(OUTPUT_DIR_CLOUDY, exist_ok=True)

# 38-Cloud directories in the new 'initial dataset' folder
BAND_DIR = 'initial dataset/train_red' 
GT_DIR = 'initial dataset/train_gt'

def extract_38cloud_tiles():
    clear_count = 0
    cloudy_count = 0
    
    gt_files = glob.glob(os.path.join(GT_DIR, '*.TIF'))
    
    if not gt_files:
        print(f"No mask files found in {GT_DIR}.")
        return 0, 0

    for gt_path in gt_files:
        if clear_count >= TARGET_SAMPLES and cloudy_count >= TARGET_SAMPLES:
            break
            
        filename = os.path.basename(gt_path)
        img_filename = filename.replace('gt', 'red') 
        img_path = os.path.join(BAND_DIR, img_filename)
        
        if not os.path.exists(img_path):
            continue
            
        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or img is None:
            continue
            
        for y in range(0, 384, TILE_SIZE):
            for x in range(0, 384, TILE_SIZE):
                
                if clear_count >= TARGET_SAMPLES and cloudy_count >= TARGET_SAMPLES:
                    return clear_count, cloudy_count
                    
                mask_patch = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]
                img_patch = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
                
                cloud_pixels = np.sum(mask_patch > 128)
                cloud_percentage = cloud_pixels / (TILE_SIZE * TILE_SIZE)
                
                if cloud_percentage > 0.60 and cloudy_count < TARGET_SAMPLES:
                    scaled_patch = (img_patch // 2).astype(np.uint8)
                    cv2.imwrite(os.path.join(OUTPUT_DIR_CLOUDY, f"tile_{cloudy_count:04d}.png"), scaled_patch)
                    cloudy_count += 1
                    
                elif cloud_percentage < 0.05 and clear_count < TARGET_SAMPLES:
                    scaled_patch = (img_patch // 2).astype(np.uint8)
                    cv2.imwrite(os.path.join(OUTPUT_DIR_CLEAR, f"tile_{clear_count:04d}.png"), scaled_patch)
                    clear_count += 1

    return clear_count, cloudy_count

if __name__ == "__main__":
    print("Starting 38-Cloud extraction...")
    final_clear, final_cloudy = extract_38cloud_tiles()
    print(f"Extraction complete. Saved {final_clear} clear tiles and {final_cloudy} cloudy tiles formatted for Vivado.")