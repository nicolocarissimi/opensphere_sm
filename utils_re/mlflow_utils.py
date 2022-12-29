from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import os

import base64
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import multiprocessing as mp

import cv2
import mlflow
from mlflow.entities.file_info import FileInfo
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts as _download_artifacts
from plotly.subplots import make_subplots
import pandas as pd
from pandas.io.formats.style import Styler


def start_run(experiment_name: str,
              run_name: str,
              description: Optional[str] = None,
              tags: Optional[Dict[str, Any]] = None) -> str:
    """Starts a new or connects to an old run.

    Args:
        experiment_name: Name of the mlflow experiment.
        run_name: Name of the run.
        description: An optional string that populates the description box of the run.
                     If a run is being resumed, the description is set on the resumed run.
                     If a new run is being created, the description is set on the new run.
        tags: An optional dictionary of string keys and values to set as tags on the run.
              If a run is being resumed, these tags are set on the resumed run.
              If a new run is being created, these tags are set on the new run.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        new_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(new_id)

    if experiment.lifecycle_stage == 'deleted':
        print(f'MLflow experiment {experiment.name} is in deleted stage. Restoring before proceeding.')
        MlflowClient().restore_experiment(experiment.experiment_id)

    run = mlflow.start_run(experiment_id=experiment.experiment_id,
                           run_name=run_name,
                           description=description,
                           tags=tags)

    return run.info.run_id


def download_artifacts(path: str, dst_path: Optional[str] = None) -> str:
    """Downloads artifact from the active run.

    Args:
        path: Relative source path to the desired artifact.
        dst_path: Absolute path of the local filesystem destination directory to which to
                  download the specified artifacts. This directory must already exist.
                  If unspecified, the artifacts will either be downloaded to a new
                  uniquely-named directory on the local filesystem or will be returned
                  directly in the case of the LocalArtifactRepository.

    Returns:
        Local path of desired artifact.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise ValueError("There is no active MLflow run.")
    return _download_artifacts(run_id=active_run.info.run_id, artifact_path=path, dst_path=dst_path)


def _cast_image(image: np.ndarray) -> str:
    """Returns a Base64 + JPEG encoded representation of `image`."""
    _, jpg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(jpg.tobytes()).decode('ascii')


def log_image_samples(images: Union[List[List[np.ndarray]], List[np.ndarray]],
                      artifact_file: str,
                      titles: Optional[List[str]] = None,
                      row_titles: Optional[List[str]] = None,
                      column_titles: Optional[List[str]] = None,
                      cols: int = 1) -> None:
    """Logs images as an HTML artifact with plotly.

    Works for limited amount of images.

    Args:
        images: list of images or list of list of images to log.
            If the former, the layout can be changed with `cols`.
            If the latter, then the layout will be used as is in `images`.
        artifact_file: filename of the artifact.
        titles: the titles of the images.
        row_titles: titles of the rows.
        column_titles: titles of the columns.
        cols: number of images per row if `images` is list of images.
    """
    if not isinstance(images[0], list):
        rows = math.ceil(len(images) / cols)
        image_layout = []
        i = 0
        for _ in range(rows):
            image_row = []
            for _ in range(cols):
                if i >= len(images):
                    break
                image_row.append(images[i])
                i = i + 1
            image_layout.append(image_row)
    else:
        image_layout = images

    rows = len(image_layout)
    cols = len(image_layout[0])
    image_height, image_width = image_layout[0][0].shape[0:2]

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, row_titles=row_titles, column_titles=column_titles)
    for row, image_row in enumerate(image_layout):
        for col, image in enumerate(image_row):
            assert image.ndim == 3 and (image.shape[2] == 3 or image.shape[2] == 1) or image.ndim == 2
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            if image.shape[2] == 1:
                image = image[:, :, (0, 0, 0)]

            fig.add_trace(go.Image(source="data:image/jpg;base64," + _cast_image(image)), row=row + 1, col=col + 1)
            fig.update_traces(row=row + 1, col=col + 1, name='')
            fig.update_xaxes(row=row + 1, col=col + 1, visible=False)
            fig.update_yaxes(row=row + 1, col=col + 1, visible=False)
    width = 1.2 * (cols * image_width) + 100
    if row_titles is not None:
        fig.update_annotations(patch=dict(textangle=0), selector=dict(textangle=90))
        fig.update_layout(margin=dict(r=150))
        width = width + 100
    fig.update_layout(autosize=False, width=width, height=1.2 * (rows * image_height) + 100)
    mlflow.log_figure(fig, artifact_file)


# def log_eval_metrics(labels: np.ndarray,
#                      predictions: np.ndarray,
#                      metrics_objs: Dict[str, tf.keras.metrics.Metric]) -> None:
#     """Logs evaluation metrics on MLflow. Predictions and labels are expected to be stored under MLflow already*.

#     * See `log_predict_images` below.

#     Args:
#         labels: Labels of the images.
#         predictions: Prediction results of the images.
#         metrics_objs: (Keras) metrics to calculate.
#     """
#     LOGGER.debug('Logging evaluation metrics...')
#     for metric_name, metric in metrics_objs.items():
#         metric.update_state(y_true=labels, y_pred=predictions)
#         LOGGER.info(f'{metric_name}: {metric.result()}')
#     mlflow.log_metrics({metric_name: metric.result().numpy() for metric_name, metric in metrics_objs.items()})
#     LOGGER.info('Evaluation metrics logged.')
