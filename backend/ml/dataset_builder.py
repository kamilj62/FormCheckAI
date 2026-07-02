"""
Dataset Builder

Converts folders of exercise videos into one-row-per-video
feature datasets using the production biomechanics pipeline.
"""

from pathlib import Path
import pandas as pd


def build_dataset(
    input_folder: str,
    label: str,
    output_csv: str,
):
    """
    Build a video-level dataset.
    """
    raise NotImplementedError


def build_multiple_datasets(config):
    """
    Build datasets from multiple folders.

    Example:

    config = {
        "clean_and_jerk": ".../raw/clean_and_jerk",
        "snatch": ".../raw/snatch_mp4",
    }
    """
    raise NotImplementedError
