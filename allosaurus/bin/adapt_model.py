import argparse
from pathlib import Path
from allosaurus.model import copy_model
from allosaurus.am.factory import transfer_am
from allosaurus.am.trainer import Trainer
from allosaurus.am.loader import read_loader

if __name__ == '__main__':

    parser = argparse.ArgumentParser("fine-tune an existing model to your target dataset")

    # required options
    parser.add_argument('--pretrained_model', required=True, type=str, help='the pretrained model id which you want to start with' )
    parser.add_argument('--new_model',        required=True, type=str, help='the new fine-tuned model id, this id show in your model list and will be available later for your inference')
    parser.add_argument('--path',             required=True, type=str, help='the data path, it should contain train directory and validate directory')
    parser.add_argument('--lang',             required=True, type=str, help='the language id of your target dataset')
    parser.add_argument('--device_id',        required=True, type=int, help='gpu cuda_device_id. use -1 if you do not have gpu')

    # non required options
    parser.add_argument('--batch_frame_size', type=int,   default=6000,  help='this indicates how many frame in each batch, if you get any memory related errors, please use a lower value for this size')
    parser.add_argument('--criterion',        type=str,   default='ctc', choices=['ctc'], help='criterion, only ctc now')
    parser.add_argument('--optimizer',        type=str,   default='sgd', choices=['sgd'], help='optimizer, only sgd now')
    parser.add_argument('--lr',               type=float, default=0.01,  help='learning rate')
    parser.add_argument('--grad_clip',        type=float, default=5.0,   help='grad clipping')
    parser.add_argument('--epoch',            type=int,   default=10,    help='number of epoch to run')
    parser.add_argument('--log',              type=str,   default='none',help='file to store training logs. do not save if none')
    parser.add_argument('--verbose',          type=bool,  default=True,  help='print all training logs on stdout')
    parser.add_argument('--report_per_batch', type=int,   default=10,    help='report training stats every N epoch')

    train_config = parser.parse_args()

    # prepare training and validating loaders
    data_path = Path(train_config.path)
    train_loader = read_loader(data_path / 'train', train_config)
    validate_loader = read_loader(data_path / 'validate', train_config)

    # initialize the target model path with the old model
    copy_model(train_config.pretrained_model, train_config.new_model)

    # setup the target model path and create model
    model = transfer_am(train_config)

    # setup trainer
    trainer = Trainer(model, train_config)

    # start training
    trainer.train(train_loader, validate_loader)

    # close datasets and loaders
    train_loader.close()
    validate_loader.close()