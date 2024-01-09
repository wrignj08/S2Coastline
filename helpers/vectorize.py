from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString
import numpy as np
from rasterio import features
import rasterio as rio
from sympy import Union
from tqdm.auto import tqdm
from shapely.geometry import shape, box
import pandas as pd

from typing import Union, Optional, List
import sys

if sys.platform == "darwin":
    from multiprocess import Pool  # type: ignore
else:
    from multiprocessing import Pool


from shapely.geometry import box
from functools import partial


def simplify_geometries(gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    new_gdf = gdf.copy()
    new_gdf["geometry"] = new_gdf["geometry"].simplify(
        tolerance, preserve_topology=False
    )
    return gpd.GeoDataFrame(new_gdf)


def get_raster_bounds(raster: Path) -> gpd.GeoDataFrame:
    with rio.open(raster) as src:
        bounds = box(*src.bounds)
        bounds_gdf = gpd.GeoDataFrame({"geometry": [bounds]})
        bounds_gdf.set_crs(src.crs, inplace=True)
        bounds_gdf = bounds_gdf.to_crs(3857)
        if bounds_gdf is not None:
            extent = bounds_gdf.geometry.values[0]
            return extent
        raise ValueError("No bounds found")


def extract_polygons(chunk: Path, px_size: float) -> gpd.GeoDataFrame:
    with rio.open(chunk) as src:
        local_epsg = src.meta["crs"].to_epsg()
        water_array = src.read(1).astype("uint8")
        mask = water_array == 1

    shapes = features.shapes(
        water_array, mask=mask, transform=src.transform, connectivity=4
    )
    water_array = None
    geoms = []
    values = []
    for shape_dict, value in shapes:
        geoms.append(shape(shape_dict))
        values.append(value)

    water_gdf = gpd.GeoDataFrame({"geometry": geoms}, crs=f"EPSG:{local_epsg}")  # type: ignore
    water_gdf = simplify_geometries(water_gdf, px_size)

    water_gdf_wgs = water_gdf.to_crs(3857)

    if not isinstance(water_gdf_wgs, gpd.GeoDataFrame):
        raise ValueError(
            "Conversion to WGS 84 CRS failed. Result is not a GeoDataFrame."
        )

    water_gdf_wgs.geometry = water_gdf_wgs.buffer(0)  # type: ignore

    return water_gdf_wgs


def chaikin_corner_cutting(points: np.ndarray, num_iterations: int = 1) -> np.ndarray:
    for _ in range(num_iterations):
        if np.array_equal(points[0], points[-1]):
            points = np.append(points, [points[1]], axis=0)

        p0 = points[:-1]
        p1 = points[1:]
        q = p0 * 0.75 + p1 * 0.25
        r = p0 * 0.25 + p1 * 0.75
        new_points = np.empty((2 * len(points) - 2, points.shape[1]))
        new_points[0::2] = q
        new_points[1::2] = r

        if np.array_equal(points[0], points[-2]):
            new_points = new_points[1:]
            new_points = np.append(new_points, [new_points[0]], axis=0)
        else:
            new_points = np.append(new_points, [points[-1]], axis=0)

        points = new_points

    return points


def smooth_gdf(gdf: gpd.GeoDataFrame, num_iterations: int = 1) -> gpd.GeoDataFrame:
    gdf["geometry"] = gdf["geometry"].apply(  # type: ignore
        lambda line: LineString(
            chaikin_corner_cutting(np.array(line.coords), num_iterations=num_iterations)
        )
    )
    return gdf


def geometry_cleanup(
    water_polygons: List[gpd.GeoDataFrame],
    coverage_gdf: gpd.GeoDataFrame,
    bounds_gdf: Union[pd.DataFrame, gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    pbar = tqdm(total=5, desc="Cleaning geometry")
    joined_water_gdf = pd.concat(water_polygons, ignore_index=True)
    pbar.set_description("Dissolving")
    joined_water_gdf_dis = joined_water_gdf.dissolve()  # type: ignore
    pbar.update(1)

    # convert multipart poly to single part so we can sort by size to remove lakes and rivers
    pbar.set_description("Exploding")
    single_part_gdf = joined_water_gdf_dis.explode(index_parts=False)
    single_part_gdf["area"] = single_part_gdf.area
    pbar.update(1)

    # only keep the largest area polygon, the ocean
    single_part_gdf = single_part_gdf.sort_values("area", ascending=False)
    single_part_gdf = single_part_gdf.iloc[[0]]

    # extract the boundary of the ocean polygon
    single_part_gdf.geometry = single_part_gdf.boundary

    # clip to remove raster edges in ocean
    pbar.set_description("Removing ocean lines")
    old_crs = bounds_gdf.crs.to_epsg()
    bounds_gdf.to_crs(3857, inplace=True)  # type: ignore
    bounds_gdf.geometry = bounds_gdf.buffer(-10)  # type: ignore
    bounds_gdf.to_crs(old_crs, inplace=True)  # type: ignore

    clipped_gdf = gpd.clip(single_part_gdf, bounds_gdf).explode(  # type: ignore
        index_parts=False  # type: ignore
    )  # type: ignore
    pbar.update(1)

    # negative buffer coverage to ensure good S2 coverage
    pbar.set_description("Removing lines based on S2 coverage")
    # coverage_gdf.geometry = coverage_gdf.buffer(-0.5)
    old_crs = coverage_gdf.crs.to_epsg()
    coverage_gdf.to_crs(3857, inplace=True)  # type: ignore
    coverage_gdf.geometry = coverage_gdf.buffer(-50000)  # type: ignore
    coverage_gdf.to_crs(old_crs, inplace=True)  # type: ignore

    # make sure coverage is in WGS84, web mercator does not work with world 'edges'
    coverage_gdf.to_crs(4326, inplace=True)

    # also reproject clipped_gdf to WGS84
    old_crs = clipped_gdf.crs.to_epsg()
    clipped_gdf.to_crs(4326, inplace=True)

    # filter out any clipped geometries that are not in the coverage area
    clipped_gdf = gpd.sjoin(
        clipped_gdf, coverage_gdf, how="inner", predicate="intersects"
    )

    clipped_gdf.to_crs(old_crs, inplace=True)
    pbar.update(1)

    pbar.set_description("Smoothing lines")
    lines_gdf = smooth_gdf(clipped_gdf, num_iterations=2)
    pbar.update(1)
    pbar.set_description("Done")
    pbar.close()

    return lines_gdf


def extract_coastline(
    input_rasters: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    s2_coverage_path = Path("data/S-2 coverage area.gpkg")
    coverage_gdf = gpd.read_file(s2_coverage_path)

    input_rasters = Path(input_rasters)

    rasters = list(input_rasters.glob("*pred.tif"))
    if not output_path:
        output_path = input_rasters.parent / f"{input_rasters.name}.gpkg"
    else:
        output_path = Path(output_path)

    raster_bounds = []
    for raster in tqdm(rasters, desc="Getting raster bounds"):
        raster_bounds.append(get_raster_bounds(raster))

    bounds_gdf = gpd.GeoDataFrame(geometry=raster_bounds)  # type: ignore
    bounds_gdf.set_crs("EPSG:3857", inplace=True)
    bounds_gdf = bounds_gdf.dissolve()

    extract_polygons_partial = partial(extract_polygons, px_size=10)
    with Pool() as p:
        water_polygons = list(
            tqdm(
                p.imap(extract_polygons_partial, rasters),
                total=len(rasters),
                desc="Extracting polygons",
            )
        )
    lines_gdf = geometry_cleanup(water_polygons, coverage_gdf, bounds_gdf)
    lines_gdf.to_file(output_path)

    return output_path
