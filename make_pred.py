import torch
import argparse
import pandas as pd
from model import LinearModel


def load_model(checkpoint_path):
    model = LinearModel(6, 4)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


def load_input_data_from_csv(csv_file):
    data = pd.read_csv(csv_file, header=None)
    input_data = torch.tensor(data.values, dtype=torch.float32)
    return input_data


def make_prediction(model, input_data):
    with torch.no_grad():
        output_probabilities = model(input_data)
    predicted_labels = output_probabilities.argmax(dim=1)
    return predicted_labels


if __name__ == "__main__":
    checkpoint_path = "experiments/best/last.pth.tar"
    parser = argparse.ArgumentParser(description="Make predictions using a pre-trained model.")
    parser.add_argument("input_csv", type=str, help="Path to the CSV file containing input data")
    args = parser.parse_args()
    model = load_model(checkpoint_path)
    input_data = load_input_data_from_csv(args.input_csv)
    predicted_labels = make_prediction(model, input_data)
    print("Predicted values: ")
    print(predicted_labels)
