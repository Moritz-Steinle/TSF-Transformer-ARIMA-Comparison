import configparser
import os
from pathlib import Path
from types import SimpleNamespace

project_root_path = Path(__file__).parent

# Get the project root directory
project_root = project_root_path

# Create a ConfigParser instance
_parser = configparser.ConfigParser()

# Read the config file
_parser.read(os.path.join(project_root, "config.ini"))
_parser.read(os.path.join(project_root, "credentials.ini"))

# Create a SimpleNamespace to hold our configuration
config = SimpleNamespace()


# Function to convert string to appropriate type
def _convert_value(value):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


# Populate the config namespace
for section in _parser.sections():
    setattr(config, section, SimpleNamespace())
    for key, value in _parser[section].items():
        setattr(getattr(config, section), key, _convert_value(value))

# Add DEFAULT section at the top level
for key, value in _parser["DEFAULT"].items():
    setattr(config, key, _convert_value(value))
