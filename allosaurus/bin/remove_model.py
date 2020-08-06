from allosaurus.model import delete_model
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('an allosaurus util to delete model')
    parser.add_argument('-m', '--model', required=True, help='model name to be deleted')

    args = parser.parse_args()
    delete_model(args.model)