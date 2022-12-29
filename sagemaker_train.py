import argparse
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Tuple

import boto3
from sagemaker import TrainingInput
from sagemaker.inputs import ShuffleConfig
from sagemaker.session import Session
from sagemaker.pytorch.estimator import PyTorch
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

SOURCE_DIR = (Path(__file__).parent).absolute()
sys.path.append(str(SOURCE_DIR))

# pylint: disable=wrong-import-position
from utils_re.settings import AWS_EXECUTION_ROLE, AWS_VPC_SUBNET, AWS_VPC_SECURITY_GROUP_ID


def setup_session_and_experiment(experiment_name: str, run_name: str,
                                 experiment_description: str) -> Tuple[Session, Optional[Dict[str, str]]]:
    """Create the local / cloud setup for the Experiments / Runs
    Args:
        cfg: Config of the Experiment
    Return:
        session: aws session or Local Session
        experiment_config : aws experiment configuration or None (for Local Mode)

    """

    # aws mode
    session = Session()
    sm = session.sagemaker_client

    classifier_experiment = Experiment.create(
        experiment_name=f"{experiment_name}-{run_name}"
                        f"-{int(time.time())}",
        description=experiment_description,
        sagemaker_boto_client=sm
    )

    trial_name = f"{run_name}-{int(time.time())}"

    Trial.create(
        trial_name=trial_name,
        experiment_name=classifier_experiment.experiment_name,
        sagemaker_boto_client=sm
    )

    experiment_config = {
        'ExperimentName': classifier_experiment.experiment_name,
        'TrialName': trial_name,
    }
    return session, experiment_config


def setup_data_source(name: str, s3_path: str) -> TrainingInput:
    """Setup for training and validation data sources. Note that no data is actually loaded here"""
    print(f"Data Location: \n{s3_path}")

    input = TrainingInput(f"{s3_path}",
                               distribution="FullyReplicated",
                               shuffle_config=ShuffleConfig(59),
                            #    content_type='application/x-tfexample',
                            #    content_type='image/jpeg',
                               content_type='text/plain',
                               s3_data_type="S3Prefix",
                               input_mode="FastFile")
                        #    input_mode="File")

    return input


def setup_hyperparams(cfg_file_path: Path) -> Dict[str, str]:
    """Setup for all SM hyper-parameters that are passed outside of the actual config"""
    # create a new path to the config file relative to sagemaker's working dir
    relative_cfg_file_path = cfg_file_path.absolute().relative_to(SOURCE_DIR)
    # Take a moment to define hyper-params and metrics (anything other than config hyperparams)
    # Notice model_dir is automatically appended to the hyper-param dict if not set to False, with a default value
    return {
        'config': str(relative_cfg_file_path)
    }


def main(cfg_file_path: Path) -> None:
    """Starts the training with a given config."""

    # - load conf file and read dataset location (read from .py config)
    with open(cfg_file_path, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)

    # - set data sources (read about fastfile/best dataloader for streaming images)
    # - set hyperparams
    # - set pytorch estimator
    #   - CUDA_VISIBLE_DEVICES
    #   - src folder
    # - run fit
    # - store model to s3

    sagemaker_session, experiment_config = setup_session_and_experiment(
                                                cfg['experiment_tracking']['experiment_name'],
                                                cfg['experiment_tracking']['run_name'],
                                                cfg['experiment_tracking']['experiment_description'])
    
    # setup data channels
    data_channels = {}
    name = cfg['data']['train']['dataset']['name']
    data_channels[name] = setup_data_source(name=name,
                                            s3_path=cfg['data']['train']['dataset']['data_dir'])
    for val_data_cfg in cfg['data']['val']:
        name = val_data_cfg['dataset']['name']
        data_channels[name] = setup_data_source(name=name,
                                                s3_path=val_data_cfg['dataset']['data_dir'])

    hyperparams = setup_hyperparams(cfg_file_path)

    # All input configurations, parameters, and metrics are automatically tracked in SM
    # Setting model_dir to False because we log everything in MLFlow
    estimator = PyTorch(
        entry_point="train.py",
        source_dir=str(SOURCE_DIR),
        model_dir=False,
        # image_uri=cfg.execution.base_image,
        framework_version='1.11.0',
        py_version='py38',
        role=AWS_EXECUTION_ROLE,
        sagemaker_session=sagemaker_session,
        instance_count=cfg['execution']['instance_count'],
        instance_type=cfg['execution']['instance_type'],
        hyperparameters=hyperparams,
        subnets=[AWS_VPC_SUBNET],
        security_group_ids=[AWS_VPC_SECURITY_GROUP_ID],
        environment={'CUDA_VISIBLE_DEVICES': '0,1,2,3'},
        # distribution={
        #     "pytorchddp": {
        #         "enabled": True
        #     }
        # }
    )

    training_job_name = f"{cfg['experiment_tracking']['experiment_name']}-{int(time.time())}"
    estimator.fit(
        inputs=data_channels,
        job_name=training_job_name,
        experiment_config=experiment_config,
        wait=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file_path',
                        help='path to the yaml config file',
                        default='config/train/vggface2_sfnet20_sphereface2.yml')
    args = parser.parse_args()

    main(Path(args.cfg_file_path).absolute())
