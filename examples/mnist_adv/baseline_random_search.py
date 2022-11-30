import torch, pickle, argparse, os, mlflow
import eagerpy as ep
from models import ConvNet
from data_loader import load_test_data
from foolbox import PyTorchModel
from attacks import LinfRandomSearch
from tqdm.notebook import tqdm
from attack_helper import batched_predictions, save_pickle
from mlflow.tracking.artifact_utils import get_artifact_uri
from functools import partialmethod


if __name__ == '__main__':
    
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=int)
    parser.add_argument('--epsilons', type=int, nargs='+', default=[0, 0.1, 0.25, 0.5, 1, 3, 5, 7.5, 10, 20, 50, 100])
    parser.add_argument('--gen_samples', type=int, default=10000)
    
    args = parser.parse_args()
    
    tracking_uri = 'sqlite:///mlruns/database.db'
    mlflow.set_tracking_uri(tracking_uri)
    df=mlflow.search_runs(experiment_names=['model_training'])
    run_id=df[df['tags.mlflow.runName']==str(args.run_name)]['run_id'].values[0]
    artifact_path = get_artifact_uri(run_id=run_id, tracking_uri=tracking_uri)
    dirs=os.listdir(artifact_path)

    for s in dirs:
        if s.find('.ckpt') >= 0:
            checkpoint = s
            break

    checkpoint_path = os.path.join(artifact_path, checkpoint)

    best_model = torch.load(checkpoint_path)
    hparams = argparse.Namespace(**best_model['hyper_parameters'])
    model = ConvNet(hparams, None, None).eval()
    model.load_state_dict(best_model['state_dict'])


    TEST_PATH = "s2_mnist_cs1.gz"
    test_data = load_test_data(TEST_PATH)
    

    images_ = test_data[:][0]
    labels_ = test_data[:][1]

    images = images_[labels_ == 0][:10]
    for i in range(1,10):
        images = torch.cat((images, images_[labels_ == i][:10]))

    del images_, labels_

    fmodel = PyTorchModel(model, bounds=(0, 255))
    clean_pred = batched_predictions(model, images, 100)
    
    epsilons = args.epsilons
    
    attack = LinfRandomSearch()
    
    success = []
    for i in range(args.gen_samples):
        *_, success_ = attack(fmodel, ep.astensor(images.cuda()), ep.astensor(clean_pred.cuda()), epsilons=epsilons)
        success.append(success_.raw.cpu())

    success = torch.stack(success).permute(1,2,0)
    success_per_sample = ep.astensor(success).float32().mean(axis=-1).raw
    success_rate = torch.mean(success_per_sample, dim=-1)
    
    save_path = os.path.join(artifact_path, attack.__class__.__name__)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    save_pickle(save_path, 'success_rate.pickle', success_rate)
    save_pickle(save_path, 'params.pickle', {'epsilons': epsilons})