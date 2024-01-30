from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio as rio
import shapely
from geopandas import GeoSeries
from pandas import DataFrame, Series
from pystac.item import Item
from pystac.item_collection import ItemCollection
from tqdm.auto import tqdm

from helpers.tide import add_tide_height


def add_bad_pixel_pct(items_df: DataFrame) -> DataFrame:
    items_df["cloud"] = items_df.apply(
        lambda row: row.iloc[0].properties["s2:high_proba_clouds_percentage"], axis=1
    )
    # also do cloud_shadow_percentage and snow_ice_percentage
    items_df["shadow"] = items_df.apply(
        lambda row: row.iloc[0].properties["s2:cloud_shadow_percentage"], axis=1
    )

    items_df["ice"] = items_df.apply(
        lambda row: row.iloc[0].properties["s2:snow_ice_percentage"], axis=1
    )
    # s2:water_percentage
    items_df["water"] = items_df.apply(
        lambda row: row.iloc[0].properties["s2:water_percentage"], axis=1
    )

    items_df["bad_pixel_pct"] = (
        items_df["ice"] + items_df["shadow"] + items_df["cloud"]
    ) - items_df["water"]

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


def download_band_pair(
    item: Series, required_bands: List[str], the_rest: DataFrame, pbar: tqdm
) -> Optional[Tuple[List[np.ndarray], List[Dict[str, Any]]]]:
    hrefs = [item.assets[band].href for band in required_bands]
    scene_bands = []
    scene_profiles = []
    try:
        with ThreadPool(2) as pool:
            bands_with_profile = pool.map(get_band, hrefs)
        for band, profile in bands_with_profile:
            scene_bands.append(band)
            scene_profiles.append(profile)
            pbar.update(1)

        pbar.refresh()
        return scene_bands, scene_profiles
    except:
        print(f"Failed to download {item}")
        # get the index of the next item from the_reset where Done == False
        if the_rest["Done"].all():
            return None
        next_item = the_rest[the_rest["Done"] == False].index[0]
        # set Done to True
        the_rest.loc[next_item, "Done"] = True
        return download_band_pair(next_item, required_bands, the_rest, pbar)


def download_bands_pool_v2(
    items_with_tide: DataFrame, time_steps: int, required_bands: List[str], pbar: tqdm
) -> Tuple[np.ndarray, Dict[str, Any]]:
    items_with_tide["Done"] = False
    top_6 = items_with_tide[:time_steps]
    the_rest = items_with_tide[time_steps:]

    partial_download_band_pair = partial(
        download_band_pair,
        required_bands=required_bands,
        the_rest=the_rest,
        pbar=pbar,
    )

    with ThreadPool(6) as pool:
        bands_and_profiles = pool.map(
            partial_download_band_pair, top_6["item"].tolist()
        )

    all_bands = []
    scene_profiles = {}
    for result in bands_and_profiles:
        if result is None:
            continue
        scene_bands, scene_profiles = result
        for band in scene_bands:
            all_bands.append(band)

    if len(all_bands) == time_steps * len(required_bands):
        return np.array(all_bands), scene_profiles[0]

    # fill missing bands with zeros
    missing_bands = time_steps * len(required_bands) - len(all_bands)
    for _ in range(missing_bands):
        all_bands.append(np.zeros_like(all_bands[0]))
    return np.array(all_bands), scene_profiles[0]


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
    """Split items by orbit and sort by no_data"""
    orbits = {}
    orbits_no_data = {}
    for item in items:
        orbit = item.properties["sat:relative_orbit"]
        no_data_pct = item.properties["s2:nodata_pixel_percentage"]
        if orbit not in orbits:
            orbits[orbit] = [item]
            orbits_no_data[orbit] = [no_data_pct]
        else:
            orbits[orbit].append(item)
            orbits_no_data[orbit].append(float(no_data_pct))
            
    # loop over orbits_no_data
    for orbit, no_data in orbits_no_data.items():
        # convert to mean of no_data
        orbits_no_data[orbit] = np.mean(no_data)
    # sort orbits by no_data
    orbits = dict(sorted(orbits.items(), key=lambda item: orbits_no_data[item[0]]))
        
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
    # print(scenes[0].properties.keys())
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

    # add scene classificaion to required bands

    all_nonzero = []
    for orbit, scenes in scenes_by_orbit.items():
        # make df from items in orbit
        items_df = pd.DataFrame(scenes)
        items_df.columns = ["item"]

        items_df = add_bad_pixel_pct(items_df)
        # sort by bad pixel pct
        items_df = items_df.sort_values(by="bad_pixel_pct", ascending=True)
        # only keep the top 20 scenes
        items_df = items_df[:20]
        items_df = add_tide_height(row.geometry.centroid, items_df, world_tides_api_key)

        # round bad pixels to nearest 10
        items_df["bad_pixel_pct"] = items_df["bad_pixel_pct"].apply(
            lambda x: round(x / 10) * 10
        )
        # Sort by cloud_pct and then by tide_height
        items_df = items_df.sort_values(
            by=["bad_pixel_pct", "tide_height"], ascending=[True, False]
        )
        # download the required bands
        bands, profile = download_bands_pool_v2(
            items_df, time_steps, required_bands, pbar
        )

        all_orbits_bands.append(bands)

        # count 0s in bands
        all_nonzero.append(np.count_nonzero(bands, axis=0))

        # sum all zeros over all orbits for each x,y
        full_nonzero_count = np.sum(np.array(all_nonzero), axis=0)
        # print(np.array(all_nonzero).shape)

        # if all x,y have at least 12 bands, we can stop
        # print(np.min(full_nonzero_count))
        if np.min(full_nonzero_count) >= band_count:
            
            # got entire scene, no need to continue
            pbar.update(band_count - pbar.n)
            return np.array(all_orbits_bands), profile
    pbar.update(band_count - pbar.n)

    return np.array(all_orbits_bands), profile


def download_row(
    row: GeoSeries,
    tide_key_path: Path,
    extract_start_year: int,
    extract_end_year: int,
    working_dir: Path,
    required_bands: List[str] = ["B03", "B08"],
    time_steps: int = 6,
    pbar: Optional[tqdm] = None,
    save_scene: bool = False,
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

    if save_scene:
        export_folder = working_dir / "scenes"
        export_folder.mkdir(exist_ok=True, parents=True)
        export_path = export_folder / (f"{row.Name}.tif")
        shape = all_orbits_bands.shape
        reshaped = all_orbits_bands.reshape(shape[0] * shape[1], shape[2], shape[3])
        export_tif(reshaped, profile, export_path)

    return all_orbits_bands, profile
