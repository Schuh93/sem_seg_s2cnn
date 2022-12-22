import torch, pickle, argparse, os, warnings, copy, time, mlflow
import numpy as np, pytorch_lightning as pl, matplotlib.pyplot as plt, eagerpy as ep
from models import ConvNet
from data_loader import load_test_data
from foolbox import PyTorchModel
from foolbox.attacks import LinfProjectedGradientDescentAttack
from foolbox.attacks.base import Repeated
from tqdm.notebook import tqdm
from attack_helper import run_batched_attack_cpu, batched_accuracy, batched_predictions, batched_predictions_eps, batched_logits_eps, save_pickle
from mlflow.tracking.artifact_utils import get_artifact_uri
from functools import partialmethod


if __name__ == '__main__':
    
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=int)
    parser.add_argument('--total', type=int, default=10000)
    parser.add_argument('--bs', type=int, default=100)
    parser.add_argument('--epsilons', type=int, nargs='+', default=[0, 0.1, 0.25, 0.5, 1, 3, 5, 7.5, 10])
    parser.add_argument('--rel_stepsize', type=float, default=0.01/0.3)
    parser.add_argument('--steps', type=int, default=70)
    parser.add_argument('--random_start', action='store_false', default=True)
    parser.add_argument('--n_repeat', type=int, default=3)
    
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


    total = args.total
    bs = args.bs

    images = test_data[:total][0]
    labels = test_data[:total][1]

    fmodel = PyTorchModel(model, bounds=(0, 255))

    clean_pred = batched_predictions(model, images, bs)


    epsilons = args.epsilons
    rel_stepsize = args.rel_stepsize
    steps = args.steps
    random_start = args.random_start
    n_repeat = args.n_repeat
    
    if not random_start:
        n_repeat = 1


    attack = Repeated(attack=LinfProjectedGradientDescentAttack(rel_stepsize=rel_stepsize, steps=steps, random_start=random_start), times=n_repeat)

    _, advs, success = run_batched_attack_cpu(attack, fmodel, images, clean_pred, epsilons, bs)
    advs = torch.stack(advs)


    success_rate = ep.astensor(success).float32().mean(axis=-1).raw
    adv_pred = batched_predictions_eps(model, advs, bs)
    logits = batched_logits_eps(model, advs, bs)


    imgs = advs[:,labels == 0][:,:10]
    for i in range(1,10):
        imgs = torch.cat((imgs, advs[:,labels == i][:,:10]), 1)

    preds = adv_pred[:,labels == 0][:,:10]
    for i in range(1,10):
        preds = torch.cat((preds, adv_pred[:,labels == i][:,:10]), 1)

    c_preds = clean_pred[labels == 0][:10]
    for i in range(1,10):
        c_preds = torch.cat((c_preds, clean_pred[labels == i][:10]))

    labs = labels[labels == 0][:10]
    for i in range(1,10):
        labs = torch.cat((labs, labels[labels == i][:10]))


    save_path = os.path.join(artifact_path, attack.attack.__class__.__name__)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    save_pickle(save_path, 'images.pickle', [imgs, preds, c_preds, labs])
    save_pickle(save_path, 'success.pickle', success)
    save_pickle(save_path, 'success_rate.pickle', success_rate)
    save_pickle(save_path, 'post_pred.pickle', adv_pred)
    save_pickle(save_path, 'params.pickle', {'epsilons': epsilons, 'rel_stepsize': rel_stepsize, 'steps': steps, 'random_start': random_start, 'n_repeat': n_repeat})
    save_pickle(save_path, 'logits.pickle', logits)