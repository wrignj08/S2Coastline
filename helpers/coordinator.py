from pathlib import Path
from threading import Thread
from typing import List, Dict, Any

from geopandas import GeoDataFrame
import numpy as np
from tqdm.auto import tqdm

# import rasterio as rio

from helpers.download import download_row
from helpers.orbit_merge import combine_and_fill
from helpers.inference import run_inference


def patch_tqdm(tqdm_instance: tqdm) -> None:
    """
    Patch the display method of a tqdm instance to handle dynamic total updates.
    This function is specifically for tqdm.notebook.tqdm instances.

    Parameters:
    tqdm_instance (tqdm or tqdm.notebook.tqdm): The tqdm instance to be patched.
    """
    # Check if the instance is from tqdm.notebook
    if hasattr(tqdm_instance, "container"):
        original_display = tqdm_instance.display

        def new_display(*args, **kwargs):
            if hasattr(tqdm_instance, "container") and tqdm_instance.total is not None:
                _, pbar, _ = tqdm_instance.container.children  # type: ignore
                pbar.max = float(tqdm_instance.total)
            original_display(*args, **kwargs)

        tqdm_instance.display = new_display
    else:
        pass


def check_path_exists(path: Path, path_name: str) -> None:
    if not path.exists():
        raise Exception(f"{path_name} {path} does not exist")


def merge_and_inf(
    bands, model_path, output_path, profile, time_steps, required_bands, inf_pbar
):
    bands = combine_and_fill(
        bands=bands,
        required_bands=required_bands,
        time_steps=time_steps,
        pbar=inf_pbar,
    )

    run_inference(
        model_path=model_path,
        output_path=output_path,
        bands=bands,
        profile=profile,
        pbar=inf_pbar,
    )


def processor(
    vector_path: Path,
    model_path: Path,
    working_dir: Path,
    tide_key_path: Path,
    extract_start_year: int,
    extract_end_year: int,
    overwrite: bool = False,
    filter_names: List[str] = [],
    time_steps: int = 6,
    required_bands: List[str] = ["B03", "B08"],
    use_tides: bool = False,
    save_scene: bool = False,
    fast_mode: bool = True,
) -> None:
    for path, name in [
        (model_path, "Model path"),
        (tide_key_path, "Tide key path"),
        (vector_path, "Vector path"),
    ]:
        check_path_exists(path, name)

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

    total_pbar = tqdm(desc="Total Progress", total=len(filter_names), position=0)
    dl_pbar = tqdm(desc="Downloading", position=1, total=10)
    inf_pbar = tqdm(desc="Waiting for download", position=2, total=10)
    patch_tqdm(dl_pbar)
    patch_tqdm(inf_pbar)

    for id, row in grid_gdf.iterrows():
        if row["Name"] not in filter_names:
            continue
        output_path = (
            working_dir
            / f"{row['Name']}_{extract_start_year}_{extract_end_year}_pred.tif"
        )
        # Skip if already exists
        if output_path.exists() and not overwrite:
            total_pbar.update(1)
            total_pbar.refresh()
            continue
        try:
            (
                bands,
                profile,
            ) = download_row(
                row,
                tide_key_path,
                extract_start_year,
                extract_end_year,
                working_dir=working_dir,
                pbar=dl_pbar,
                time_steps=time_steps,
                required_bands=required_bands,
                save_scene=save_scene,
                use_tides=use_tides,
            )
            if inf_thread.is_alive():
                inf_thread.join()
            if bands is None:
                failed.append(row["Name"])
                print(f"Failed on {row['Name']}")
                total_pbar.update(1)
                continue

            inf_thread = Thread(
                target=merge_and_inf,
                args=(
                    bands,
                    model_path,
                    output_path,
                    profile,
                    time_steps,
                    required_bands,
                    inf_pbar,
                ),
            )

            inf_thread.start()
            if not fast_mode:
                inf_thread.join()
            total_pbar.update(1)

        except Exception as e:
            print(e)
            failed.append(row["Name"])
            print(f"Failed on {row['Name']}")
            total_pbar.update(1)
            continue
    # hold until all inference threads are finished
    if inf_thread.is_alive():
        inf_thread.join()
    total_pbar.close()
