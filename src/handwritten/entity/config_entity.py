import os
import sys
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass




@dataclass(frozen=True)
class DataIngestionConfig:

    train_data_path  : Path
    test_data_path   : Path
    val_data_path    : Path
    raw_data_path    : Path 