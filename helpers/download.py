from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio as rio
import shapely
from geopandas import GeoSeries
from pandas import DataFrame
from pystac.item import Item
from pystac.item_collection import ItemCollection
from tqdm.auto import tqdm

from helpers.tide import add_tide_height


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
            pbar.refresh()
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
    pbar: tqdm,
    band_count: int = 12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    profile = {}

    pbar.reset()
    pbar.total = len(scenes_by_orbit) * band_count
    pbar.set_description(f"Downloading {row.Name}")
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

        # check if any 0s in bands
        if np.count_nonzero(bands) == bands.size:
            # got entire scene, no need to continue
            all_orbits_bands.append(bands)
            pbar.update(band_count - pbar.n)
            return np.array(all_orbits_bands), profile
        all_orbits_bands.append(bands)

    return np.array(all_orbits_bands), profile


def download_row(
    row: GeoSeries,
    tide_key_path: Path,
    extract_start_year: int,
    extract_end_year: int,
    required_bands: List[str] = ["B03", "B08"],
    time_steps: int = 6,
    pbar: Optional[tqdm] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if pbar is None:
        pbar = tqdm(leave=False)

    world_tides_api_key = tide_key_path.read_text().strip()

    scenes = get_scenes(row, extract_start_year, extract_end_year)

    if len(scenes) == 0:
        raise Exception(f"No scenes found for {row.Name}")

    scenes_by_orbit = split_by_orbits(scenes)

    all_orbits_bands, profile = download_each_orbit(
        scenes_by_orbit, row, world_tides_api_key, time_steps, required_bands, pbar
    )

    return all_orbits_bands, profile
