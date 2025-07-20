import yaml
import os


from paths import CONFIG_FILE_PATH, DATA_DIR


def load_config(config_path: str = CONFIG_FILE_PATH):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
