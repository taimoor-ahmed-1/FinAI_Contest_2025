import numpy as np
import pandas as pd

def fix_input_data():
    """
    Fix NaN values in the input data by replacing them with appropriate values.
    """
    
    print("Fixing NaN values in input data...")
    
    # Load the input data
    input_path = "./data/BTC_1sec_input_train_70.npy"
    data = np.load(input_path)
    
    print(f"Original shape: {data.shape}")
    print(f"Original NaN count: {np.isnan(data).sum()}")
    
    # Convert to float32 for better precision
    data = data.astype(np.float32)
    
    # Replace NaN values with 0 (simple approach)
    # For time series data, this is often a reasonable default
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"After fixing - NaN count: {np.isnan(data).sum()}")
    print(f"Min: {data.min():.6f}, Max: {data.max():.6f}, Mean: {data.mean():.6f}")
    
    # Save the fixed data
    np.save(input_path, data)
    print(f"Fixed data saved to: {input_path}")
    
    # Also fix validation and test data
    for split in ["val", "test_15"]:
        input_path = f"./data/BTC_1sec_input_{split}.npy"
        try:
            data = np.load(input_path)
            print(f"\nFixing {split} data...")
            print(f"Original NaN count: {np.isnan(data).sum()}")
            
            data = data.astype(np.float32)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"After fixing - NaN count: {np.isnan(data).sum()}")
            np.save(input_path, data)
            print(f"Fixed {split} data saved")
        except FileNotFoundError:
            print(f"File {input_path} not found, skipping...")

if __name__ == "__main__":
    fix_input_data() 