import json
import os
from pathlib import Path
import requests
from tqdm import tqdm


def download_eurorad_figures(metadata_path: str, output_dir: str) -> None:
    """
    Download figures from Eurorad dataset and save them organized by case_id.

    Args:
        metadata_path: Path to the eurorad_metadata.json file
        output_dir: Base directory where figures will be saved

    The figures will be saved as:
        {output_dir}/{case_id}/{figure_number}.jpg
    Example:
        figures/189/Figure_1a.jpg
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Iterate through all cases with progress bar
    for case_id in tqdm(metadata, desc="Downloading cases", unit="case"):
        case = metadata[case_id]
        case_dir = output_path / str(case["case_id"])
        case_dir.mkdir(exist_ok=True)

        # Process all figures and their subfigures
        for figure in case["figures"]:
            for subfig in figure["subfigures"]:

                # Remove leading and trailing whitespace and convert to lowercase
                subfig_name = f"{subfig['number'].strip().replace(' ', '_').lower()}.jpg"
                subfig_path = Path(case_dir) / subfig_name

                save_figure(
                    url=subfig["url"],
                    output_path=subfig_path,
                )


def save_figure(url: str, output_path: Path) -> None:
    """
    Download and save a single figure.

    Args:
        url: URL of the figure to download
        output_path: Path where the figure should be saved
    """
    if output_path.exists():
        return

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"Error downloading {url}: {e}")


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    download_eurorad_figures(
        metadata_path=os.path.join(root, "eurorad_metadata.json"),
        output_dir=os.path.join(root, "figures"),
    )
