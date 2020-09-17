from pathlib import Path
import shutil

def get_all_models():
    """
    get all local models

    :return:
    """
    model_dir = Path(__file__).parent / 'pretrained'
    models = list(sorted(model_dir.glob('*'), reverse=True))

    #assert len(models) > 0, "No models are available, you can maually download a model with download command or just run inference to download the latest one automatically"

    return models

def get_model_path(model_name):
    """
    get model path by name, verify its a valid path

    :param model_name: str
    :return: model path
    """

    model_dir = Path(__file__).parent / 'pretrained'

    resolved_model_name = resolve_model_name(model_name)

    assert resolved_model_name != "none", model_name+" is not a valid model name. please check by list_model"

    return model_dir / resolved_model_name

def copy_model(src_model_name, tgt_model_name):
    """
    copy a model to a new model

    :param src_model_name:
    :param tgt_model_name:
    :return:
    """

    # verify the source path is not empty
    src_model_path = get_model_path(src_model_name)

    # verify the target path is empty
    model_dir = Path(__file__).parent / 'pretrained'
    tgt_model_path = model_dir / tgt_model_name

    assert not tgt_model_path.exists(), \
        "provided model name "+tgt_model_name+" has already exist. Consider another name or delete the existing one"

    shutil.copytree(str(src_model_path), str(tgt_model_path))

def delete_model(model_name):

    model_path = get_model_path(model_name)

    answer = input(f"you will delete {model_path}? [Y|N]")
    if answer.lower() in ['y', 'yes', 'true']:
        print("deleting ", model_path)
        shutil.rmtree(str(model_path))


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