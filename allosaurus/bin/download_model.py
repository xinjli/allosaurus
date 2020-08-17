from pathlib import Path
import tarfile
from urllib.request import urlopen
import io
import argparse
import os

def download_model(model_name=None):

    if model_name is None:
        model_name = 'latest'

    model_dir = (Path(__file__).parent.parent) / 'pretrained'

    if not (model_dir / model_name).exists():

        try:
            url = 'https://www.pyspeech.com/static/model/recognition/allosaurus/' + model_name + '.tar.gz'
            print("downloading model ", model_name)
            print("from: ", url)
            print("to:   ", str(model_dir))
            print("please wait...")
            resp = urlopen(url)
            compressed_files = io.BytesIO(resp.read())
            files = tarfile.open(fileobj=compressed_files)
            files.extractall(str(model_dir))

        except:
            print("Error: could not download the model")
            (model_dir / model_name).rmdir()


if __name__ == '__main__':

    parser = argparse.ArgumentParser('a utility to download pretrained models')
    parser.add_argument('-m', '--model', default='latest',  help='specify which model to download. A list of downloadable models are available on Github')

    args = parser.parse_args()

    download_model(args.model)