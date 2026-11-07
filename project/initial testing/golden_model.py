import numpy as np

def run_golden_model():
    #(act_in, weight_in, psum_in)
    test_cases = [
        ("Zero Case",      0,    0,    0),
        ("Standard Case", 15,   -3,   100),
        ("Max Positive",  127,  127,  2147483600), # Pushed near the 32-bit limit
        ("Max Negative", -128, -128,  0)           # -128 * -128 = +16384
    ]

    print("--- PE Golden Model Results ---")
    
    for name, act, weight, psum in test_cases:
        
        act_np = np.int8(act)
        weight_np = np.int8(weight)
        psum_in_np = np.int32(psum)
        
       
        # Upcast to int32 before multiplying to prevent 8-bit overflow
        product = np.int32(act_np) * np.int32(weight_np)
        
        # Final accumulation
        psum_out_np = np.int32(product + psum_in_np)
        
        print(f"Test: {name}")
        print(f"  Inputs : act={act_np}, weight={weight_np}, psum_in={psum_in_np}")
        print(f"  Product: {product}")
        print(f"  Output : psum_out = {psum_out_np}")
        # Print Hex for easy Vivado waveform comparison
        print(f"  Hex Out: 0x{int(psum_out_np) & 0xFFFFFFFF:08X}\n")

if __name__ == "__main__":
    run_golden_model()