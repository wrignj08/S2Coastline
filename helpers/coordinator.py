from pathlib import Path
from threading import Thread
from tqdm.auto import tqdm
from geopandas import GeoDataFrame
from typing import List
from helpers.inference import run_inference
from helpers.download import download_row


def from_vector(
    vector_path: Path,
    model_path: Path,
    working_dir: Path,
    tide_key_path: Path,
    extract_start_year: int,
    extract_end_year: int,
    overwrite: bool = False,
    filter_names: List[str] = [],
) -> None:
    if not model_path.exists():
        raise Exception(f"Model path {model_path} does not exist")

    if not tide_key_path.exists():
        raise Exception(f"Tide key path {tide_key_path} does not exist")

    if not vector_path.exists():
        raise Exception(f"Vector path {vector_path} does not exist")

    if not working_dir.exists():
        print(f"Working directory {working_dir} does not exist, creating")
        working_dir.mkdir(exist_ok=True, parents=True)

    print("Loading grid")
    grid_gdf = GeoDataFrame.from_file(vector_path)

    # Set filter names
    if filter_names == []:
        filter_names = grid_gdf["Name"].tolist()
        print("No filter names provided, using all names")
    else:
        print("Filtering by names")

    failed = []
    inf_thread = Thread()
    for id, row in tqdm(grid_gdf.iterrows(), total=len(filter_names)):
        if row["Name"] not in filter_names:
            continue
        export_path = (
            working_dir
            / f"{row['Name']}_{extract_start_year}_{extract_end_year}_pred.tif"
        )
        # Skip if already exists
        if export_path.exists() and not overwrite:
            continue
        try:
            bands, profile = download_row(
                row,
                tide_key_path,
                extract_start_year,
                extract_end_year,
            )
            if inf_thread.is_alive():
                inf_thread.join()
            if bands is None:
                failed.append(id)
                print(f"Failed on {id}")
                continue
            inf_thread = Thread(
                target=run_inference,
                args=(
                    model_path,
                    export_path,
                    bands,
                    profile,
                ),
            )
            inf_thread.start()

        except Exception as e:
            print(e)
            failed.append(id)
            print(f"Failed on {id}")
            continue
