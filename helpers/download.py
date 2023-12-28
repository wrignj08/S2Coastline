from typing import Any, Dict, List, Optional, Tuple
from pandas import DataFrame
import planetary_computer
import pystac_client
import numpy as np
import shapely
import pandas as pd
import rasterio as rio
from tqdm.auto import tqdm
from helpers.tide import add_tide_height, setup_database


def add_cloud_pct(items_df: DataFrame) -> DataFrame:
    items_df["cloud_pct"] = items_df.apply(
        lambda row: row.iloc[0].properties["eo:cloud_cover"], axis=1
    )
    return items_df


def get_band(
    href: str, attempt: int = 0
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
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
            print(f"Failed to open {href} after 3 attempts")
            return None, None


def download_bands(items_with_tide, time_steps, required_bands):
    bands = []

    profile = {}
    pbar = tqdm(total=time_steps * len(required_bands), leave=False)
    for id, row in items_with_tide.iterrows():
        scene_bands = []

        for band in required_bands:
            href = row["item"].assets[band].href
            band, profile = get_band(href)
            if type(band) == type(None):
                print(f"Failed to download {href}")
                scene_bands = []
                break
            pbar.update(1)

            scene_bands.append(band)
        for band in scene_bands:
            bands.append(band)
        if len(bands) == time_steps * len(required_bands):
            return bands, profile
    return bands, profile


def split_by_orbits(items):
    orbits = {}
    for item in items:
        orbit = item.properties["sat:relative_orbit"]
        if orbit not in orbits:
            orbits[orbit] = [item]
        else:
            orbits[orbit].append(item)
    return orbits


def export_tif(array, profile, export_path):
    profile.update(count=array.shape[0])
    with rio.open(export_path, "w", **profile) as dst:
        dst.write(array)


def get_scenes(row, extract_start_year, extract_end_year):
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
    scenes_by_orbit, row, world_tides_api_key, time_steps, required_bands
):
    all_orbits_bands = []
    profile = {}
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
        bands, profile = download_bands(items_df, time_steps, required_bands)
        all_orbits_bands.append(bands)

    return all_orbits_bands, profile


def combine_orbits(all_orbits_bands, target_bands):
    all_orbits_bands = np.array(all_orbits_bands)
    all_orbits_bands = np.moveaxis(all_orbits_bands, 0, 1)

    out_shape = (target_bands, *all_orbits_bands.shape[2:])
    out_array = np.zeros(out_shape, dtype=np.uint16)

    for index, multi_orbit_bands in enumerate(all_orbits_bands):
        target_array = np.zeros(multi_orbit_bands.shape[1:])
        for band in multi_orbit_bands:
            target_array[target_array == 0] = band[target_array == 0]
        out_array[index] = target_array

    return out_array


def download_row(row, tide_key_path, extract_start_year, extract_end_year):
    world_tides_api_key = tide_key_path.read_text().strip()

    required_bands = ["B03", "B08"]
    target_bands = 12
    time_steps = 6

    scenes = get_scenes(row, extract_start_year, extract_end_year)

    if len(scenes) == 0:
        return

    scenes_by_orbit = split_by_orbits(scenes)

    all_orbits_bands, profile = download_each_orbit(
        scenes_by_orbit,
        row,
        world_tides_api_key,
        time_steps,
        required_bands,
    )

    out_array = combine_orbits(all_orbits_bands, target_bands)

    del all_orbits_bands

    if out_array.shape[0] != target_bands:
        print(f"Failed to download {row.Name}")
        return None

    return out_array, profile


def make_fake_data(*args):
    """
    Generate fake data array and profile.

    Returns:
        array (numpy.ndarray): Fake data array.
        profile (dict): Profile containing metadata for the array.
    """
    array = np.random.randint(0, 10000, (12, 10980, 10980)).astype(np.uint16)
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": None,
        "width": 10980,
        "height": 10980,
        "count": 12,
        "crs": "EPSG:32630",
        "transform": rio.transform.from_origin(499980.0, 4500210.0, 10.0, 10.0),  # type: ignore
    }
    return array, profile
