from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config_Data:
    batch_size: int
    num_epochs: int
    lr: float
    seq_len: int
    d_model: int
    datasource: str
    lang_src: str
    lang_tgt: str
    model_folder: str
    model_basename: str
    preload: str
    tokenizer_file: str
    experiment_name: str

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    num_epochs: int
    lr: float
    seq_len: int
    d_model: int