import torch, argparse, gzip, os, warnings, copy, time, mlflow, pickle
import numpy as np, pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.tracking.artifact_utils import get_artifact_uri, _get_root_uri_and_artifact_path
from data_loader import load_test_data
from models import S2ConvNet
from mlflow_helper import init_mlf_logger




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--channels', type=int, nargs='+')
    parser.add_argument('--bandlimit', type=int, nargs='+')
    parser.add_argument('--kernel_max_beta', type=float, nargs='+')
    parser.add_argument('--activation_fn', type=str, default='ReLU')
    parser.add_argument('--batch_norm', action='store_true', default=True)
    parser.add_argument('--nodes', type=int, nargs='+') 
    parser.add_argument('--test_rot', action='store_true', default=True)
    
    args = parser.parse_args()
    
    hparams = argparse.Namespace()
    hparams.name = args.name
    hparams.test_batch_size = args.test_batch_size
    hparams.num_workers = args.num_workers
    hparams.channels = args.channels
    hparams.bandlimit = args.bandlimit
    hparams.kernel_max_beta = args.kernel_max_beta
    hparams.activation_fn = args.activation_fn
    hparams.batch_norm = args.batch_norm
    hparams.nodes = args.nodes
    hparams.lr = 1e-3
    hparams.weight_decay = 0

    if args.test_rot:
        test_path = "s2_mnist_cs1.gz"
    else:
        raise NotImplementedError('A non-rotated test set does not exist yet.')
    
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found.')
    
    test_data = load_test_data(test_path)
    
    tracking_uri='sqlite:///mlruns/database.db'

    tag_dict = {"mlflow.runName": round(time.time()),
               "mlflow.user": "dschuh"}

    mlf_logger, artifact_path = init_mlf_logger(experiment_name='model_training', tracking_uri=tracking_uri, tags=tag_dict)
    
    model = S2ConvNet(hparams, None, test_data)
    mlf_logger.experiment.set_tag(run_id=mlf_logger.run_id, key="model", value=model.__class__.__name__+'_u')
    
    print(f"Number of total parameters: {model.count_parameters()}")

    monitor = 'val_acc'
    mode = 'max'
    checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint(filepath=artifact_path, monitor=monitor, mode=mode)
    
    log_dict = {'test_rot': args.test_rot,
               'total_params': model.count_parameters()}

    mlf_logger.log_hyperparams(log_dict)

    trainer = pl.Trainer(gpus=1, logger=mlf_logger, checkpoint_callback=checkpoint)
    
    checkpoint_dict = {'state_dict': copy.deepcopy(model.state_dict()),
                      'hyper_parameters': pl.utilities.parsing.AttributeDict(vars(hparams))
                      }

    assert not os.path.isfile(os.path.join(artifact_path, 'untrained.ckpt'))
    torch.save(checkpoint_dict, os.path.join(artifact_path, 'untrained.ckpt'))
    
    model.eval()
    test_results = trainer.test(model)
    
    filename = 'test_results.pickle'
    if os.path.isfile(os.path.join(artifact_path, filename)):
        filename = str(time.time()) + filename
        print('File already existed, timestamp was prepended to filename.')

    with open(os.path.join(artifact_path, filename), 'wb') as file:
        pickle.dump(test_results, file)