import os
import sys
from seq_data import ConfigData, convert_btc_csv_to_btc_npy

def generate_data_for_splits():
    """
    Generate processed data (Alpha101 factors and labels) for all split files.
    This ensures proper train/test separation without look-ahead bias.
    """
    
    # Use the 70/15/15 split that was actually created
    splits = [
        ("train", "70% training data"),
        ("val", "15% validation data"), 
        ("test", "15% test data")
    ]
    
    for split_type, description in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split_type.upper()} split ({description})...")
        print(f"{'='*50}")
        
        try:
            # Create ConfigData for this split
            args = ConfigData(split_type=split_type)
            
            # Check if the CSV file exists
            if not os.path.exists(args.csv_path):
                print(f"Warning: {args.csv_path} not found. Skipping {split_type} split.")
                continue
            
            print(f"Input CSV: {args.csv_path}")
            print(f"Output files:")
            print(f"  - Input features: {args.input_ary_path}")
            print(f"  - Labels: {args.label_ary_path}")
            print(f"  - Predictions: {args.predict_ary_path}")
            
            # Generate the data
            convert_btc_csv_to_btc_npy(args=args)
            
            print(f"✓ Successfully processed {split_type} split")
            
        except Exception as e:
            print(f"✗ Error processing {split_type} split: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Data generation complete!")
    print(f"{'='*50}")
    
    # Print summary of generated files
    print("\nGenerated files:")
    for split_type, description in splits:
        args = ConfigData(split_type=split_type)
        for file_path in [args.input_ary_path, args.label_ary_path, args.predict_ary_path]:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✓ {file_path} ({size_mb:.1f} MB)")
            else:
                print(f"  ✗ {file_path} (missing)")

if __name__ == "__main__":
    generate_data_for_splits() 