from typing import Any, Dict, List, Tuple
from pandas import DataFrame
import planetary_computer
import pystac_client
import numpy as np
import shapely
import pandas as pd
import rasterio as rio
from tqdm.auto import tqdm
from helpers.tide import add_tide_height
from geopandas import GeoSeries
from typing import List, Dict, Any
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pathlib import Path
from multiprocessing.pool import ThreadPool


def add_cloud_pct(items_df: DataFrame) -> DataFrame:
    items_df["cloud_pct"] = items_df.apply(
        lambda row: row.iloc[0].properties["eo:cloud_cover"], axis=1
    )
    return items_df


def get_band(href: str, attempt: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        singed_href = planetary_computer.sign(href)
        with rio.open(singed_href) as src:
            return src.read(1), src.profile.copy()
    except Exception as e:
        print(e)
        print(f"Failed to open {href}")
        if attempt < 3:
            print(f"Trying again {attempt+1}")
            return get_band(href, attempt + 1)
        else:
            raise Exception(f"Failed to open {href}")


def download_bands_pool(
    items_with_tide: DataFrame, time_steps: int, required_bands: List[str], pbar: tqdm
) -> Tuple[np.ndarray, Dict[str, Any]]:
    bands = []
    profile = {}

    for item in items_with_tide["item"].tolist():
        hrefs = [item.assets[band].href for band in required_bands]

        try:
            with ThreadPool(2) as pool:
                bands_with_profile = pool.map(get_band, hrefs)
            for band, profile in bands_with_profile:
                bands.append(band)
                pbar.update(1)
        except:
            print(f"Failed to download {item}")
            continue

        if len(bands) == time_steps * len(required_bands):
            return np.array(bands), profile

    # fill missing bands with zeros
    missing_bands = time_steps * len(required_bands) - len(bands)
    for _ in range(missing_bands):
        bands.append(np.zeros_like(bands[0]))
    return np.array(bands), profile


def split_by_orbits(items: ItemCollection) -> Dict[str, List[Item]]:
    orbits = {}
    for item in items:
        orbit = item.properties["sat:relative_orbit"]
        if orbit not in orbits:
            orbits[orbit] = [item]
        else:
            orbits[orbit].append(item)
    return orbits


def export_tif(array: np.ndarray, profile: Dict[str, Any], export_path: Path) -> None:
    profile.update(count=array.shape[0], dtype=array.dtype, nodata=None)
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)


def get_scenes(
    row: GeoSeries, extract_start_year: int, extract_end_year: int
) -> ItemCollection:
    bounds = row.geometry.buffer(-0.05)

    query = {
        "collections": ["sentinel-2-l2a"],
        "intersects": shapely.to_geojson(bounds),
        "datetime": f"{extract_start_year}-01-01T00:00:00Z/{extract_end_year}-12-31T23:59:59Z",
        "query": {"s2:mgrs_tile": {"eq": row.Name}},
    }
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
    )
    scenes = catalog.search(**query).item_collection()
    return scenes


def download_each_orbit(
    scenes_by_orbit: Dict[str, List[Item]],
    row: GeoSeries,
    world_tides_api_key: str,
    time_steps: int,
    required_bands: List[str],
    band_count: int = 12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = {}
    pbar = tqdm(
        total=len(scenes_by_orbit) * band_count, leave=False, desc="Downloading"
    )
    all_orbits_bands = []

    for orbit, scenes in scenes_by_orbit.items():
        # make df from items in orbit
        items_df = pd.DataFrame(scenes)
        items_df.columns = ["item"]

        items_df = add_cloud_pct(items_df)
        # sort by cloud cover
        items_df = items_df.sort_values(by="cloud_pct", ascending=True)
        # only keep the top 20 scenes
        items_df = items_df[:20]
        items_df = add_tide_height(row.geometry.centroid, items_df, world_tides_api_key)
        # round tide height to nearest 10
        items_df["cloud_pct"] = items_df["cloud_pct"].apply(
            lambda x: round(x / 10) * 10
        )
        # Sort by cloud_pct and then by tide_height
        items_df = items_df.sort_values(
            by=["cloud_pct", "tide_height"], ascending=[True, False]
        )
        # download the required bands
        bands, profile = download_bands_pool(items_df, time_steps, required_bands, pbar)
        all_orbits_bands.append(bands)

    return np.array(all_orbits_bands), profile


def combine_orbits(all_orbits_bands: np.ndarray, target_bands: int) -> np.ndarray:
    all_orbits_bands = np.moveaxis(all_orbits_bands, 0, 1)

    out_shape = (target_bands, *all_orbits_bands.shape[2:])
    out_array = np.zeros(out_shape, dtype=np.uint16)

    for index, multi_orbit_bands in enumerate(all_orbits_bands):
        target_array = np.zeros(multi_orbit_bands.shape[1:])
        for band in multi_orbit_bands:
            target_array[target_array == 0] = band[target_array == 0]
        out_array[index] = target_array

    return out_array


def fill_missing_data(combined_arrays: np.ndarray, time_steps: int) -> np.ndarray:
    """
    Fills missing data in bands of a combined array using subsequent bands.
    """
    target_bands = combined_arrays.shape[0]
    channels = target_bands // time_steps
    filled_array = np.zeros_like(combined_arrays)

    for channel in range(channels):
        one_channel = combined_arrays[channel::channels]
        fall_back_values = np.zeros_like(one_channel[0])
        # sort one_channel index 0 by non zero values
        # so that we can fill in missing data with the next band
        non_zero_counts = np.array([np.count_nonzero(slice) for slice in one_channel])
        sorted_indices = np.argsort(-non_zero_counts)

        for band_id in sorted_indices:
            band = one_channel[band_id]
            fall_back_values[fall_back_values == 0] = band[fall_back_values == 0]

        for id, band in enumerate(one_channel):
            band[band == 0] = fall_back_values[band == 0]
            one_channel[id] = band
        filled_array[channel::channels] = one_channel

    return filled_array


def download_row(
    row: GeoSeries,
    tide_key_path: Path,
    extract_start_year: int,
    extract_end_year: int,
    required_bands: List[str] = ["B03", "B08"],
    target_bands: int = 12,
    time_steps: int = 6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    world_tides_api_key = tide_key_path.read_text().strip()

    scenes = get_scenes(row, extract_start_year, extract_end_year)

    if len(scenes) == 0:
        raise Exception(f"No scenes found for {row.Name}")

    scenes_by_orbit = split_by_orbits(scenes)

    all_orbits_bands, profile = download_each_orbit(
        scenes_by_orbit,
        row,
        world_tides_api_key,
        time_steps,
        required_bands,
    )

    combined_arrays = combine_orbits(all_orbits_bands, target_bands)

    combined_arrays = fill_missing_data(combined_arrays, time_steps)

    del all_orbits_bands

    if combined_arrays.shape[0] != target_bands:
        raise Exception(f"Failed to download {row.Name}")

    return combined_arrays, profile
