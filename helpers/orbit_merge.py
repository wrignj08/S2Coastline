from typing import List

import numpy as np
from tqdm.auto import tqdm


def combine_orbits(
    all_orbits_bands: np.ndarray, target_band_count: int, pbar: tqdm
) -> np.ndarray:
    """
    Combines multiple orbits of bands into a single array.
    """
    all_orbits_bands = np.moveaxis(all_orbits_bands, 0, 1)

    out_shape = (target_band_count, *all_orbits_bands.shape[2:])
    out_array = np.zeros(out_shape, dtype=np.uint16)
    pbar.reset()
    pbar.set_description(f"Combining")
    pbar.total = len(all_orbits_bands)

    for index, multi_orbit_bands in enumerate(all_orbits_bands):
        target_array = np.zeros(multi_orbit_bands.shape[1:])
        for band in multi_orbit_bands:
            target_array[target_array == 0] = band[target_array == 0]
        out_array[index] = target_array
        pbar.update(1)

    return out_array


def fill_missing_data(
    combined_arrays: np.ndarray, time_steps: int, pbar: tqdm
) -> np.ndarray:
    """
    Fills missing data in bands of a combined array using subsequent bands.
    """
    target_bands = combined_arrays.shape[0]
    channels = target_bands // time_steps
    filled_array = np.zeros_like(combined_arrays)

    pbar.reset()
    pbar.set_description(f"Filling")
    pbar.total = channels

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
        pbar.update(1)

    return filled_array


def combine_and_fill(
    bands: np.ndarray, required_bands: List[str], time_steps: int, pbar: tqdm
) -> np.ndarray:
    """
    Combines multiple orbits of bands into a single array and fills missing data.
    """
    target_band_count = time_steps * len(required_bands)
    combined_arrays = combine_orbits(bands, target_band_count, pbar)

    filled_array = fill_missing_data(combined_arrays, time_steps, pbar)
    return filled_array
