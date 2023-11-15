import torch, argparse, gzip, os, warnings, copy, time, mlflow, pickle
import numpy as np, pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.tracking.artifact_utils import get_artifact_uri, _get_root_uri_and_artifact_path
from data_loader import load_train_data, load_test_data
from models import ConvNet, CConvNet
from mlflow_helper import init_mlf_logger




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--channels', type=int, nargs='+')
    parser.add_argument('--kernels', type=int, nargs='+')
    parser.add_argument('--strides', type=int, nargs='+')
    parser.add_argument('--activation_fn', type=str, default='ReLU')
    parser.add_argument('--batch_norm', action='store_false', default=True)
    parser.add_argument('--nodes', type=int, nargs='+')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--train_samples', type=int, default=6e4)
    parser.add_argument('--flat', action='store_true', default=False)
    parser.add_argument('--padded_img_size', type=int, nargs='+', default=[60,60])
    parser.add_argument('--train_rot', action='store_false', default=True)
    parser.add_argument('--test_rot', action='store_false', default=True)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--min_delta', type=float, default=0.)
    parser.add_argument('--patience', type=int, default=10)
    
    args = parser.parse_args()
    
    hparams = argparse.Namespace()
    hparams.name = args.name
    hparams.train_batch_size = args.train_batch_size
    hparams.test_batch_size = args.test_batch_size
    hparams.num_workers = args.num_workers
    hparams.lr = args.lr
    hparams.weight_decay = args.weight_decay
    hparams.channels = args.channels
    hparams.kernels = args.kernels
    hparams.strides = args.strides
    hparams.activation_fn = args.activation_fn
    hparams.batch_norm = args.batch_norm
    hparams.nodes = args.nodes
    
    if args.flat:
        if args.train_rot:
            train_path = "flat_mnist_train_aug_" + str(args.padded_img_size[0]) + "x" + str(args.padded_img_size[1]) + "_" +  str(args.train_samples) + ".gz"
        else:
            train_path = "flat_mnist_train_" + str(args.padded_img_size[0]) + "x" + str(args.padded_img_size[1]) + "_" +  str(args.train_samples) + ".gz"
        
        if args.test_rot:
            test_path = "flat_mnist_test_aug_" + str(args.padded_img_size[0]) + "x" + str(args.padded_img_size[1]) + ".gz"
        else:
            test_path = "flat_mnist_test_" + str(args.padded_img_size[0]) + "x" + str(args.padded_img_size[1]) + ".gz"
    else:
        if args.train_rot:
            train_path = "s2_mnist_train_dwr_" + str(args.train_samples) + ".gz"
        else:
            train_path = "s2_mnist_train_sphere_center_" + str(args.train_samples) + ".gz"

        if args.test_rot:
            test_path = "s2_mnist_cs1.gz"
        else:
            test_path = "s2_mnist_test_sphere_center.gz"
    
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found.')
    
    train_data = load_train_data(train_path)
    if args.test_rot and not args.flat:
        test_data = load_test_data(test_path)
    else:
        test_data = load_train_data(test_path)
    
    tracking_uri='sqlite:///mlruns/database.db'

    tag_dict = {"mlflow.runName": round(time.time()),
               "mlflow.user": "dschuh"}

    mlf_logger, artifact_path = init_mlf_logger(experiment_name='model_training', tracking_uri=tracking_uri, tags=tag_dict)
    
    if args.flat and args.train_rot:
        model = ConvNet(hparams, train_data, test_data)
    else:
        model = CConvNet(hparams, train_data, test_data)
    mlf_logger.experiment.set_tag(run_id=mlf_logger.run_id, key="model", value=model.__class__.__name__)

    print(f"Number of trainable / total parameters: {model.count_trainable_parameters(), model.count_parameters()}")

    monitor = 'val_acc'
    mode = 'max'
    early_stopping = pl.callbacks.EarlyStopping(monitor=monitor, min_delta=args.min_delta, patience=args.patience, mode=mode)
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=artifact_path, monitor=monitor, mode=mode)

    log_dict = {'es_min_delta': early_stopping.min_delta,
               'es_mode': early_stopping.mode,
               'es_monitor': early_stopping.monitor,
               'es_patience': early_stopping.patience,
               'max_epochs': args.max_epochs,
               'train_samples': len(train_data),
               'flat': args.flat,
               'padded_img_size': args.padded_img_size,
               'train_rot': args.train_rot,
               'test_rot': args.test_rot,
               'trainable_params': model.count_trainable_parameters(),
               'total_params': model.count_parameters()}
    
    if not args.flat:
        del log_dict['padded_img_size']

    mlf_logger.log_hyperparams(log_dict)

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs, logger=mlf_logger, early_stop_callback=early_stopping, checkpoint_callback=checkpoint)

    trainer.fit(model)

    mlf_logger.experiment.log_param(run_id=mlf_logger.run_id, key='es_stopped_epoch', value=early_stopping.stopped_epoch)

    best_model = torch.load(checkpoint.best_model_path)
    model.load_state_dict(best_model['state_dict'])
    model.eval()
    test_results = trainer.test(model)
    
    filename = 'test_results.pickle'
    if os.path.isfile(os.path.join(artifact_path, filename)):
        filename = str(time.time()) + filename
        print('File already existed, timestamp was prepended to filename.')

    with open(os.path.join(artifact_path, filename), 'wb') as file:
        pickle.dump(test_results, file)