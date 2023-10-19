import torch, pickle, argparse, os, mlflow
import eagerpy as ep
from models import ConvNet, CConvNet
from data_loader import load_test_data, load_train_data
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
    parser.add_argument('--mode', type=str, default='mean')
    
    args = parser.parse_args()
    
    assert args.mode in ['mean', 'max'], "Supported modes are 'min' and 'max'."
    
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
    if df[df['tags.mlflow.runName']==str(args.run_name)]['tags.model'].values[0] == 'ConvNet':
        model = ConvNet(hparams, None, None).eval()
    elif df[df['tags.mlflow.runName']==str(args.run_name)]['tags.model'].values[0] == 'CConvNet':
        model = CConvNet(hparams, None, None).eval()
    else:
        raise NotImplementedError(f"Model has to be 'ConvNet' or 'CConvNet'. Got {df[df['tags.mlflow.runName']==str(args.run_name)]['tags.model'].values[0]}.")
        
    model.load_state_dict(best_model['state_dict'])


    test_rot = eval(df[df['tags.mlflow.runName']==str(args.run_name)]['params.test_rot'].values[0])
    
    if df[df['tags.mlflow.runName']==str(args.run_name)]['params.flat'].values[0] is None:
        flat = False
    else:
        flat = eval(df[df['tags.mlflow.runName']==str(args.run_name)]['params.flat'].values[0])
    
    if flat:
        padded_img_size = eval(df[df['tags.mlflow.runName']==str(args.run_name)]['params.padded_img_size'].values[0])
        
        if test_rot:
            TEST_PATH = "flat_mnist_test_aug_" + str(padded_img_size[0]) + "x" + str(padded_img_size[1]) + ".gz"
        else:
            TEST_PATH = "flat_mnist_test_" + str(padded_img_size[0]) + "x" + str(padded_img_size[1]) + ".gz"
        
        test_data = load_train_data(TEST_PATH)
        
    else:    
        if test_rot:
            TEST_PATH = "s2_mnist_cs1.gz"
            test_data = load_test_data(TEST_PATH)
        else:
            TEST_PATH = "s2_mnist_test_sphere_center.gz"
            test_data = load_train_data(TEST_PATH)
    

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
    if args.mode == 'mean':
        success_per_sample = ep.astensor(success).float32().mean(axis=-1).raw
        name = attack.__class__.__name__
    else:
        success_per_sample = torch.amax(ep.astensor(success).float32().raw, dim=-1)
        name = attack.__class__.__name__ + '_max'
    
    success_rate = torch.mean(success_per_sample, dim=-1)
    save_path = os.path.join(artifact_path, name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    save_pickle(save_path, 'success_rate.pickle', success_rate)
    save_pickle(save_path, 'params.pickle', {'epsilons': epsilons})