from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from strictyaml import YAML, load

# Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PROJECT_ROOT / "config.yml"

class Cameras(BaseModel):
    # cam_garaz: str
    # cam_taras: str
    # cam_wejscie: str
    # cam_maly_taras: str
    cam_schody: str
    cam_salon: str

class ModelConfig(BaseModel):
    yolo_config_file: str
    yolo_weights_file: str
    coco_names_file: str

class Config(BaseModel):
    model_config: ModelConfig
    cam_config: Cameras


def find_config_file() -> Path:
    """locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at: {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration"""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    else:
        raise OSError(f"Did not find config file at path: {cfg_path}")
    
def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values"""

    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictaml YAML type
    _config = Config(
        cam_config=Cameras(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )
    return _config


config = create_and_validate_config()