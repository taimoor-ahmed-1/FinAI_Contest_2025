class ConfigData:
    def __init__(self, data_dir: str = "./data", split_type: str = "train"):
        self.data_dir = data_dir
        
        # Use split files based on split_type (70/15/15 split)
        if split_type == "train":
            # Use the 70% split file
            self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_train_70.csv"
            self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"
        elif split_type == "test":
            # Use the 15% test split file
            self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_test_15.csv"
            self.predict_ary_path = f"{data_dir}/BTC_1sec_predict_test_15.npy"
        elif split_type == "val":
            # Use the 15% validation split file
            self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_val.csv"
            self.predict_ary_path = f"{data_dir}/BTC_1sec_predict_val.npy"
        else:
            # Fallback to original file
            self.csv_path = f"{data_dir}/BTC_1sec_with_sentiment_risk_train.csv"
            self.predict_ary_path = f"{data_dir}/BTC_1sec_predict.npy"