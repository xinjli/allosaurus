from allosaurus.model import get_all_models


if __name__ == '__main__':

    models = get_all_models()

    if len(models) == 0:
        print("No models are available, you can maually download a model with download command or just run inference to download the latest one automatically")
    else:
        print("Available Models")
        for i, model in enumerate(models):
            if i == 0:
                print('-', model.name, "(default)")
            else:
                print('-', model.name)