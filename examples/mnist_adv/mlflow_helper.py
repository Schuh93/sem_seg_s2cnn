import os, mlflow, pickle
from mlflow.tracking.artifact_utils import get_artifact_uri, _get_root_uri_and_artifact_path
from pytorch_lightning.loggers import MLFlowLogger


def create_artifact_dir(artifact_path, mlf_logger, verbose=False):
    if not os.path.isdir(os.path.join('./mlruns', mlf_logger.experiment_id)):
        os.mkdir(os.path.join('./mlruns', mlf_logger.experiment_id))
        if verbose:
            print('experiment ID directory created')

    if not os.path.isdir(_get_root_uri_and_artifact_path(artifact_path)[0]):
        os.mkdir(_get_root_uri_and_artifact_path(artifact_path)[0])
        if verbose:
            print('run ID directory created')

    if not os.path.isdir(artifact_path):
        os.mkdir(artifact_path)
        if verbose:
            print('artifact directory created')
        
def init_mlf_logger(experiment_name, tracking_uri, tags=None, verbose=False):
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=tracking_uri, tags=tags)
    artifact_path = get_artifact_uri(run_id=mlf_logger.run_id, tracking_uri=tracking_uri)
    create_artifact_dir(artifact_path, mlf_logger, verbose=verbose)
    return mlf_logger, artifact_path


def load_attack_results(run_name: int, attack: str, filename: str):
    tracking_uri = 'sqlite:///mlruns/database.db'
    mlflow.set_tracking_uri(tracking_uri)
    df=mlflow.search_runs(experiment_names=['model_training'])
    run_id=df[df['tags.mlflow.runName']==str(run_name)]['run_id'].values[0]
    artifact_path = get_artifact_uri(run_id=run_id, tracking_uri=tracking_uri)
    attack_path = os.path.join(artifact_path, attack)
    with open(os.path.join(attack_path, filename), 'rb') as file:
        data = pickle.load(file)
    return data


def load_from_db(run_name: int, param: str):
    tracking_uri = 'sqlite:///mlruns/database.db'
    mlflow.set_tracking_uri(tracking_uri)
    df=mlflow.search_runs(experiment_names=['model_training'])
    return df[df['tags.mlflow.runName']==str(run_name)][param].values[0]