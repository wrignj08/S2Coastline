import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

import requests
from pandas import DataFrame
from shapely.geometry import Point
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By


def setup_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS tide_data
                    (date TEXT, latitude REAL, longitude REAL, tide_height REAL)"""
    )
    conn.commit()
    return conn


def query_tide_data(cursor, date, latitude, longitude) -> Optional[float]:
    """
    Query the local database for tide data.

    Args:
        cursor: The database cursor object.
        date: The date for which tide data is being queried.
        latitude: The latitude of the location for which tide data is being queried.
        longitude: The longitude of the location for which tide data is being queried.

    Returns:
        The tide height for the specified date, latitude, and longitude, or None if no data is found.
    """
    cursor.execute(
        "SELECT tide_height FROM tide_data WHERE date = ? AND latitude = ? AND longitude = ?",
        (date, latitude, longitude),
    )
    result = cursor.fetchone()
    return result[0] if result else None


def store_tide_data(
    cursor: sqlite3.Cursor,
    date: str,
    latitude: float,
    longitude: float,
    tide_height: float,
    conn: sqlite3.Connection,
) -> None:
    """Store new tide data in the local database."""
    cursor.execute(
        "INSERT INTO tide_data (date, latitude, longitude, tide_height) VALUES (?, ?, ?, ?)",
        (date, latitude, longitude, tide_height),
    )

    conn.commit()


def add_tide_height(
    centroid: Point, items_df: DataFrame, world_tides_api_key: str
) -> DataFrame:
    """
    Adds tide height information to the items dataframe based on the centroid coordinates and datetime information.

    Args:
        centroid (Point): The centroid coordinates.
        items_df (DataFrame): The dataframe containing the items.
        world_tides_api_key (str): The API key for accessing the World Tides API.

    Returns:
        DataFrame: The updated items dataframe with tide height information.
    """
    db_path = Path("tide_data.db")
    conn = setup_database(db_path)
    cursor = conn.cursor()

    results = []
    lon, lat = centroid.coords[0]
    for id, item in items_df.iterrows():
        dt_str = item.iloc[0].to_dict()["properties"]["datetime"]
        dt_obj = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        date = dt_obj.date().isoformat()

        # Check the database first
        tide_height = query_tide_data(cursor, date, lat, lon)
        # If not in the database, fetch from API and store
        if tide_height is None:
            url = f"https://www.worldtides.info/api/v3?heights&date={date}&lat={lat}&lon={lon}&key={world_tides_api_key}"
            response = requests.get(url)
            data = json.loads(response.text)

            min_diff = float("inf")
            closest_entry = {}
            target_timestamp = dt_obj.timestamp()
            try:
                data["heights"]
            except KeyError:
                print(f"No tide data found setting all tide heights to 0")
                # fill with 0 and return
                tide_height = 0
                items_df["tide_height"] = tide_height
                return items_df
                # continue

            for entry in data["heights"]:
                diff = abs(entry["dt"] - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_entry = entry

            tide_height = closest_entry["height"]
            store_tide_data(cursor, date, lat, lon, tide_height, conn)

        results.append(tide_height)

    items_df["tide_height"] = results

    conn.close()

    return items_df


def get_worldtides_credit(creds_json: Optional[Path] = None) -> Optional[int]:
    if creds_json is None:
        creds_json = Path("world_tide_creds.json")

    with open(creds_json) as f:
        creds = json.load(f)
        email = creds["email"]
        password = creds["password"]
    # Setup the Chrome WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to the login page
        driver.get("https://www.worldtides.info/login")

        driver.implicitly_wait(
            10
        )  # Wait up to 10 seconds for the elements to become available

        # Locate the email and password fields and input the creds
        email_field = driver.find_element(
            By.XPATH, '//input[@placeholder="Email address"]'
        )
        password_field = driver.find_element(
            By.XPATH, '//input[@placeholder="Password"]'
        )

        email_field.send_keys(email)
        password_field.send_keys(password)

        # Submit
        password_field.send_keys(Keys.RETURN)

        # Navigate to overview
        driver.get("https://www.worldtides.info/overview")

        driver.implicitly_wait(10)

        # Find the div containing <h4>Prepaid</h4> and then find the <h3> within this div
        prepaid_div = driver.find_element(
            By.XPATH, '//h4[text()="Prepaid"]/ancestor::div[h1]'
        )
        h1_value = prepaid_div.find_element(By.TAG_NAME, "h1").text

        driver.quit()

        return int(h1_value)

    except Exception as e:
        print(f"Error: {e}")
        driver.quit()
