import torch
import argparse


def print_model_info(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))

    print(f"Keys in the model '{model_path}':")
    for key in model.keys():
        print(key)

    print("\nTensor shapes in the model:")
    for key, value in model.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Tensor shape: {value.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Path to the model')
    args = parser.parse_args()

    print_model_info(args.model)