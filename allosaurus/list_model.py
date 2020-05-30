from pathlib import Path

def get_all_models():
    """
    get all local models

    :return:
    """
    model_dir = Path(__file__).parent / 'pretrained'
    models = list(sorted(model_dir.glob('*'), reverse=True))
    return models

def resolve_model_name(model_name='latest'):
    """
    select the model

    :param model_name:
    :return:
    """

    models = get_all_models()

    # get the latest model in local
    if model_name == 'latest':
        return models[0].name

    # match model name
    for model in models:
        if model.name == model_name:
            return model_name

    return "none"


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