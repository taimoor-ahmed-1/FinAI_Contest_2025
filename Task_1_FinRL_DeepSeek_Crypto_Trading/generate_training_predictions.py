import os
import sys
import torch as th
import numpy as np
from seq_data import ConfigData
from seq_net import RnnRegNet

def generate_training_predictions():
    """
    Generate RNN predictions for the training split (70% data).
    This is needed for RL training as the TradeSimulator expects these predictions.
    """
    
    print("Generating RNN predictions for training split...")
    
    # Load training data
    train_args = ConfigData(split_type="train")
    
    # Load the trained RNN model
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    net = RnnRegNet(
        inp_dim=101,  # Alpha101 features
        mid_dim=128, 
        out_dim=8,    # 8 price predictions
        num_layers=4
    ).to(device)
    
    # Load trained weights
    model_path = "./data/BTC_1sec_predict.pth"
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return
    
    net.load_state_dict(th.load(model_path, map_location=device))
    net.eval()
    
    # Load training input features
    input_path = train_args.input_ary_path
    if not os.path.exists(input_path):
        print(f"Error: Training input features not found at {input_path}")
        return
    
    input_seq = np.load(input_path)
    input_seq = th.tensor(input_seq, dtype=th.float32, device=device)
    
    print(f"Input shape: {input_seq.shape}")
    
    # Generate predictions
    predict_ary = np.empty((input_seq.shape[0], 8))
    hid = None
    seq_len = 2 ** 8  # 256
    
    print(f"Generating predictions for {input_seq.shape[0]} time steps...")
    
    for seq_i0 in range(0, input_seq.shape[0], seq_len):
        seq_i1 = min(seq_i0 + seq_len, input_seq.shape[0])
        inp = input_seq[seq_i0:seq_i1].unsqueeze(1)  # Add batch dimension
        
        with th.no_grad():
            out, hid = net.forward(inp, hid)
        
        predict_ary[seq_i0:seq_i1] = out.squeeze(1).cpu().numpy()
        
        if seq_i0 % 10000 == 0:
            print(f"Processed {seq_i0}/{input_seq.shape[0]} steps")
    
    # Save predictions
    output_path = "./data/BTC_1sec_predict.npy"
    np.save(output_path, predict_ary)
    print(f"Training predictions saved to: {output_path}")
    print(f"Prediction shape: {predict_ary.shape}")

if __name__ == "__main__":
    generate_training_predictions() 