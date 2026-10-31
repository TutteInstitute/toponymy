from pathlib import Path
import json


def example_ids() -> list[str]:
    """
    Return a list of available example IDs for prompt examples.

    Returns
    -------
    list[str]
        A list of example identifiers (e.g., ["hockey_masks"]).
    """
    return ["hockey_masks"]


def prompt_example_params_path(example_id: str) -> Path:
    """
    Get the path to the JSON file containing prompt parameters for a given example.

    Parameters
    ----------
    example_id : str
        The identifier for the example (e.g., "hockey_masks").

    Returns
    -------
    Path
        A Path object pointing to the JSON file with prompt parameters.
    """
    file_path = (
        Path(__file__).parent.parent.parent
        / "examples"
        / "prompt_example_params"
        / f"{example_id}.json"
    )
    if not file_path.exists():
        raise FileNotFoundError(f"Prompt parameters file not found: {file_path}")
    return file_path


def load_prompt_params(example_id: str) -> dict:
    """
    Load prompt parameters from a JSON file for a given example.

    Parameters
    ----------
    example_id : str
        The identifier for the example (e.g., "hockey_masks").

    Returns
    -------
    dict
        A dictionary containing the prompt parameters.
    """
    file_path = prompt_example_params_path(example_id)
    with open(file_path, "r") as f:
        params = json.load(f)
    return params
