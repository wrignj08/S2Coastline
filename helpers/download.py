import requests
import planetary_computer
import pystac_client
import numpy as np
import requests
import json
from datetime import datetime
import shapely
import pandas as pd
import rasterio as rio
from tqdm.auto import tqdm
import sqlite3


def add_cloud_pct(items_df):
    items_df["cloud_pct"] = items_df.apply(
        lambda row: row.iloc[0].properties["eo:cloud_cover"], axis=1
    )
    return items_df


def setup_database(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS tide_data
                        (date TEXT, latitude REAL, longitude REAL, tide_height REAL)"""
        )
        conn.commit()
        return conn
    except Exception as e:
        print(f"Error setting up database: {e}")
        return None


def query_tide_data(cursor, date, latitude, longitude):
    """Query the local database for tide data."""
    cursor.execute(
        "SELECT tide_height FROM tide_data WHERE date = ? AND latitude = ? AND longitude = ?",
        (date, latitude, longitude),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def store_tide_data(cursor, date, latitude, longitude, tide_height, conn):
    """Store new tide data in the local database."""
    cursor.execute(
        "INSERT INTO tide_data (date, latitude, longitude, tide_height) VALUES (?, ?, ?, ?)",
        (date, latitude, longitude, tide_height),
    )
    conn.commit()


def add_tide_height(cursor, conn, centroid, items_df, world_tides_api_key):
    results = []
    lon, lat = centroid.coords[0]
    for id, item in items_df.iterrows():
        dt_str = item.iloc[0].to_dict()["properties"]["datetime"]
        dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        date = dt_obj.date().isoformat()

        # Check the database first
        tide_height = query_tide_data(cursor, date, lat, lon)
        if tide_height is None:
            # If not in the database, fetch from API and store
            url = f"https://www.worldtides.info/api/v3?heights&date={date}&lat={lat}&lon={lon}&key={world_tides_api_key}"
            response = requests.get(url)
            data = json.loads(response.text)

            min_diff = float("inf")
            closest_entry = {}
            target_timestamp = dt_obj.timestamp()
            for entry in data["heights"]:
                diff = abs(entry["dt"] - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_entry = entry

            tide_height = closest_entry["height"]
            store_tide_data(cursor, date, lat, lon, tide_height, conn)

        results.append(tide_height)

    items_df["tide_height"] = results
    return items_df


def get_band(href, attempt=0):
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


def download_scene(
    row,
    target_bands,
    time_steps,
    extract_start_year,
    extract_end_year,
    required_bands,
    world_tides_api_key,
):
    db_path = "tide_data.db"
    conn = setup_database(db_path)

    if conn is None:
        raise Exception("Failed to set up the database.")
    cursor = conn.cursor()

    centroid = row.geometry.centroid
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

    if len(scenes) == 0:
        return

    scenes_by_orbit = split_by_orbits(scenes)
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
        items_df = add_tide_height(
            cursor, conn, centroid, items_df, world_tides_api_key
        )
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

    all_orbits_bands = np.array(all_orbits_bands)
    all_orbits_bands = np.moveaxis(all_orbits_bands, 0, 1)

    out_shape = (target_bands, *all_orbits_bands.shape[2:])
    out_array = np.zeros(out_shape, dtype=np.uint16)
    for index, multi_orbit_bands in enumerate(all_orbits_bands):
        target_array = np.zeros(multi_orbit_bands.shape[1:])
        for band in multi_orbit_bands:
            target_array[target_array == 0] = band[target_array == 0]
        out_array[index] = target_array
    del all_orbits_bands

    if out_array.shape[0] != target_bands:
        print(f"Failed to download {row.Name}")
        return None

    conn.close()

    return out_array, profile


def download_row(row, tide_key_path, extract_start_year, extract_end_year):
    world_tides_api_key = tide_key_path.read_text().strip()

    required_bands = ["B03", "B08"]
    target_bands = 12
    time_steps = 6

    return download_scene(
        row,
        target_bands,
        time_steps,
        extract_start_year,
        extract_end_year,
        required_bands,
        world_tides_api_key,
    )
