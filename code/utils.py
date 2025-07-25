import yaml
import os
import json


from paths import CONFIG_FILE_PATH, DATA_DIR


def load_config(config_path: str = CONFIG_FILE_PATH):
    """Load the configuration file.

    Args:
        config_path (str, optional): Path to the config file. Defaults to CONFIG_FILE_PATH.

    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(filepath: str):
    """
    Reads a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def read_yaml(config_path: str):
    """
    Reads a YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Loaded configuration.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(filepath: str):
    """
    Reads a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def load_publication_example(example_number: int) -> str:
    """
    Load a publication example text file.

    Args:
        example_number: The number of the example to load (1, 2, or 3)

    Returns:
        The content of the publication example file
    """
    example_fpath = f"publication_example{example_number}.md"
    full_path = os.path.join(DATA_DIR, example_fpath)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def load_toxic_example() -> str:
    """
    Load a toxic example text file.

    Returns:
        The content of the toxic example file as a string.
    """
    example_path = os.path.join(DATA_DIR, "toxic_example.md")
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read()


def load_unusual_prompt_example() -> str:
    """
    Load a unusual prompt example text file.

    Returns:
        The content of the unusual prompt example file as a string.
    """
    example_path = os.path.join(DATA_DIR, "unusual_prompt_example.md")
    with open(example_path, "r", encoding="utf-8") as f:
        return f.read()
