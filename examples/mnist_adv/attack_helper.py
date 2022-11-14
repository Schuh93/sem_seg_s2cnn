import torch, os, pickle
import eagerpy as ep, numpy as np
from typing import Tuple, List
from tqdm.notebook import tqdm
from foolbox import accuracy



def run_batched_attack(attack, fmodel, images, labels, epsilons, bs) -> Tuple[List[ep.Tensor], List[ep.Tensor], ep.Tensor]:
    assert len(images) == len(labels)
    n_samples = len(labels)
    assert n_samples % bs == 0, f'The batch size ({bs}) must be a divisor of number of samples ({n_samples}).'
    with tqdm(total=n_samples//bs) as pbar:
        raw_advs, clipped_advs, success = attack(fmodel, ep.astensor(images[:bs].cuda()), ep.astensor(labels[:bs].cuda()), epsilons=epsilons)
        pbar.update(1)

        for i in range(1,n_samples//bs):
            dummy_raw_advs, dummy_clipped_advs, dummy_success = attack(fmodel, ep.astensor(images[bs*i:bs*(i+1)].cuda()), ep.astensor(labels[bs*i:bs*(i+1)].cuda()), epsilons=epsilons)

            for j in range(len(epsilons)):
                raw_advs[j] = ep.concatenate((raw_advs[j], dummy_raw_advs[j]), axis=0)
                clipped_advs[j] = ep.concatenate((clipped_advs[j], dummy_clipped_advs[j]), axis=0)
            success = ep.concatenate((success, dummy_success), axis=1)
            pbar.update(1)
            
    return raw_advs, clipped_advs, success


def run_batched_attack_cpu(attack, fmodel, images, labels, epsilons, bs) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    assert len(images) == len(labels)
    n_samples = len(labels)
    assert n_samples % bs == 0, f'The batch size ({bs}) must be a divisor of number of samples ({n_samples}).'
    with tqdm(total=n_samples//bs) as pbar:
        raw_advs, clipped_advs, success = attack(fmodel, ep.astensor(images[:bs].cuda()), ep.astensor(labels[:bs].cuda()), epsilons=epsilons)
        raw_advs_cpu = [raw_advs[i].raw.cpu() for i in range(len(raw_advs))]
        clipped_advs_cpu = [clipped_advs[i].raw.cpu() for i in range(len(clipped_advs))]
        success_cpu = success.raw.cpu()
        pbar.update(1)

        for i in range(1,n_samples//bs):
            raw_advs, clipped_advs, success = attack(fmodel, ep.astensor(images[bs*i:bs*(i+1)].cuda()), ep.astensor(labels[bs*i:bs*(i+1)].cuda()), epsilons=epsilons)
            dummy_raw_advs_cpu = [raw_advs[i].raw.cpu() for i in range(len(raw_advs))]
            dummy_clipped_advs_cpu = [clipped_advs[i].raw.cpu() for i in range(len(clipped_advs))]
            dummy_success_cpu = success.raw.cpu()

            for j in range(len(epsilons)):
                raw_advs_cpu[j] = torch.cat((raw_advs_cpu[j], dummy_raw_advs_cpu[j]), axis=0)
                clipped_advs_cpu[j] = torch.cat((clipped_advs_cpu[j], dummy_clipped_advs_cpu[j]), axis=0)
            success_cpu = torch.cat((success_cpu, dummy_success_cpu), axis=1)
            pbar.update(1)
            
    return raw_advs_cpu, clipped_advs_cpu, success_cpu


def batched_accuracy(fmodel, images, labels, bs) -> np.float64:
    assert len(images) == len(labels)
    n_samples = len(labels)
    assert n_samples % bs == 0, f'The batch size ({bs}) must be a divisor of number of samples ({n_samples}).'
    clean_accuracy = []
    for i in tqdm(range(n_samples//bs)):
        clean_accuracy.append(accuracy(fmodel, ep.astensor(images[bs*i:bs*(i+1)].cuda()), ep.astensor(labels[bs*i:bs*(i+1)].cuda())))
        
    return np.mean(clean_accuracy)


def batched_predictions(model, images, bs):
    n_samples = len(images)
    assert n_samples % bs == 0, f'The batch size ({bs}) must be a divisor of number of samples ({n_samples}).'
    
    clean_pred = []
    for i in tqdm(range(n_samples//bs)):
        output = model(images[bs*i:bs*(i+1)].cuda())
        clean_pred.append(torch.max(output, 1)[1].cpu())
        
    return torch.stack(clean_pred).flatten()


def batched_predictions_eps(model, advs, bs):
    return torch.stack([batched_predictions(model, advs[i], bs) for i in range(len(advs))])


def batched_logits(model, images, bs):
    n_samples = len(images)
    assert n_samples % bs == 0, f'The batch size ({bs}) must be a divisor of number of samples ({n_samples}).'
    
    logits = []
    for i in tqdm(range(n_samples//bs)):
        logits.append(model(images[bs*i:bs*(i+1)].cuda()).detach().cpu())
        
    return torch.stack(logits).reshape(-1,10)


def batched_logits_eps(model, advs, bs):
    return torch.stack([batched_logits(model, advs[i], bs) for i in range(len(advs))])


def save_pickle(save_path, filename, data):
    if os.path.isfile(os.path.join(save_path, filename)):
        filename = str(time.time()) + filename
        print('File already existed, timestamp was prepended to filename.')
    
    with open(os.path.join(save_path, filename), 'wb') as file:
        pickle.dump(data, file)