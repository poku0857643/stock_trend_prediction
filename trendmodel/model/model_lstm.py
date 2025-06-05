import torch
import torch.nn as nn
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm1.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        lstm_out, (h_n, _) = self.lstm1(x)
        last_hidden = h_n[-1]

        if last_hidden.size(0) > 1:
            last_hidden = self.batch_norm(last_hidden)

        last_hidden = self.dropout(last_hidden)

        last_hidden = torch.clamp(last_hidden, min=-10, max=10)
        output = self.fc1(last_hidden)
        output = self.fc1(last_hidden)
        output = torch.clamp(output, min=-1e6, max=1e6)

        if torch.isnan(output).any():
            print("Warning: NaN detected is output, replacing with zeros")
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        return output


def predict_next_day(df, features_for_scaler, features_for_model, model, scaler, target_index_in_scaler, n_steps=60):
# Use only the latest n_steps data
    df = df.copy()
    df[features_for_scaler] = df[features_for_scaler].astype(float)

    if len(df) < n_steps:
        raise ValueError("Not enough data to make prediction.")

    # all features scaled
    recent_data = df[features_for_scaler].values[-n_steps:]
    scaled_input = scaler.transform(recent_data)

    # select scaled features for train without target feature "close"
    model_input = scaled_input[:, [features_for_scaler.index(f) for f in features_for_model]]
    x = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)  # shape (1, n_steps, n_features)

    # Predict
    with torch.no_grad():
        pred = model(x).numpy().flatten()[0]

    # Inverse transform assuming 'close' is last column
    padded = np.zeros((1, len(features_for_scaler)))
    padded[0, target_index_in_scaler] = pred
    predicted_close = scaler.inverse_transform(padded)[0, target_index_in_scaler]
    return predicted_close




def load_model_2(model_path, feature_dim):
    # Create the model architecture
    model = LSTMModel(
        input_size=feature_dim,
        output_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    )
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model