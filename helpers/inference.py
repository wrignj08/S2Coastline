import gc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio as rio
import torch
from torch import Tensor
from tqdm.auto import tqdm

BAND_IDS = [2, 4]
MEANS = np.array(
    [
        0.09320524,
        0.09936677,
        0.09359581,
        0.09989304,
        0.09392498,
        0.0994415,
        0.09318926,
        0.09834657,
        0.09105494,
        0.09607462,
        0.09178863,
        0.09679132,
    ]
)
STDS = np.array(
    [
        0.02172433,
        0.02760383,
        0.02274428,
        0.02833729,
        0.02223172,
        0.0276719,
        0.02222958,
        0.02731097,
        0.02183141,
        0.02698776,
        0.02132447,
        0.02619315,
    ]
)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore
        return torch.device("mps")
    return torch.device("cpu")


def normalize(band_stack: torch.Tensor, device: torch.device) -> torch.Tensor:
    means = np.tile(MEANS[BAND_IDS], 6)  # type: ignore
    stds = np.tile(STDS[BAND_IDS], 6)  # type: ignore

    means_tensor = Tensor(means).view(1, -1, 1, 1).to(device).half()
    stds_tensor = Tensor(stds).view(1, -1, 1, 1).to(device).half()

    return (band_stack / 32767 - means_tensor) / stds_tensor


def create_gradient_mask(patch_size: int, patch_overlap_px: int) -> np.ndarray:
    if patch_overlap_px > 0:
        gradient_strength = 1
        gradient = np.ones((patch_size, patch_size), dtype=int) * patch_overlap_px
        gradient[:, :patch_overlap_px] = np.tile(
            np.arange(1, patch_overlap_px + 1),
            (patch_size, 1),
        )
        gradient[:, -patch_overlap_px:] = np.tile(
            np.arange(patch_overlap_px, 0, -1),
            (patch_size, 1),
        )
        gradient = gradient / patch_overlap_px
        rotated_gradient = np.rot90(gradient)
        combined_gradient = rotated_gradient * gradient

        combined_gradient = (combined_gradient * gradient_strength) + (
            1 - gradient_strength
        )
    else:
        combined_gradient = np.ones((patch_size, patch_size), dtype=int)
    return combined_gradient


def make_patches(
    band_stack: np.ndarray,
    pbar: Optional[tqdm],
    patch_size: int,
    overlap: int = 20,
    scene_size: int = 10980,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    patches, locations = [], []
    row_count = (scene_size - overlap) // (patch_size - overlap)

    if pbar is None:
        pbar = tqdm(leave=False)
    pbar.reset()
    pbar.total = row_count
    pbar.set_description("Making patches")

    top = 0
    while top < scene_size:
        left = 0
        while left < scene_size:
            adjusted_top = min(top, scene_size - patch_size)
            adjusted_left = min(left, scene_size - patch_size)
            patch = band_stack[
                :,
                adjusted_top : adjusted_top + patch_size,
                adjusted_left : adjusted_left + patch_size,
            ]
            patches.append(patch)
            locations.append((adjusted_top, adjusted_left))
            left += patch_size - overlap

        top += patch_size - overlap
        pbar.update(1)

    pbar.update(row_count - pbar.n)

    return patches, locations


def stitch_preds(
    preds: np.ndarray,
    locations: List[Tuple[int, int]],
    overlap: int = 20,
    scene_size: int = 10980,
    pbar: Optional[tqdm] = None,
) -> np.ndarray:
    if pbar is None:
        pbar = tqdm(leave=False)
    pbar.reset()
    pbar.total = len(preds)
    pbar.set_description("Stitching")

    gradient = create_gradient_mask(preds[0].shape[-1], overlap)
    pred_array = np.zeros((scene_size, scene_size))
    count_tracker = np.zeros((scene_size, scene_size))

    for pred, location in zip(preds, locations):
        top, left = location
        pred_array[top : top + pred.shape[-1], left : left + pred.shape[-1]] += (
            pred[0] * gradient
        )

        count_tracker[
            top : top + pred.shape[-1], left : left + pred.shape[-1]
        ] += gradient
        pbar.update(1)

    pred_array = pred_array / count_tracker

    return pred_array


def export_pred(
    output_path: Path,
    pred_array: np.ndarray,
    profile: Dict[str, Any],
    binary: bool = True,
    pbar: Optional[tqdm] = None,
) -> None:
    if pbar is None:
        pbar = tqdm(leave=False)
    pbar.reset()
    pbar.total = 1
    pbar.set_description("Exporting")
    profile["nodata"] = None
    if binary:
        profile.update(dtype=rio.int8, count=1, compress="lzw", driver="GTiff")
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(pred_array > 0, 1)
    else:
        profile.update(dtype=rio.float32, count=1, compress="lzw", driver="GTiff")
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(pred_array, 1)
    pbar.update(1)


def inference(
    patches: List[np.ndarray],
    model,
    device: torch.device,
    batch_size: int = 10,
    roll_tta_depth: int = 6,
    rotate_tta_depth: int = 2,
    pbar: Optional[tqdm] = None,
) -> np.ndarray:
    all_preds = []
    batch_count = len(patches) // batch_size
    if pbar is None:
        pbar = tqdm(leave=False)

    pbar.reset()
    pbar.total = batch_count * roll_tta_depth * rotate_tta_depth
    pbar.set_description(f"Inference on {device.type}")

    model.eval()
    with torch.no_grad():
        for batch_id in range(batch_count):
            batch = patches[batch_id * batch_size : (batch_id + 1) * batch_size]
            batch = np.array(batch).astype(np.float32)

            batch = torch.tensor(batch).to(device).half()
            batch = normalize(batch, device)

            tta_preds = []
            # apply TTA rotations
            for rotate_tta in range(rotate_tta_depth):
                batch = torch.rot90(batch, rotate_tta, [2, 3])
                # apply TTA rolls
                for roll_tta in range(roll_tta_depth):
                    # creating augmented batch for TTA
                    rolled_batch = torch.roll(batch, shifts=2 * roll_tta, dims=1)

                    pred = model(rolled_batch)
                    # remove TTA rotation
                    pred = torch.rot90(pred, -rotate_tta, [2, 3])

                    tta_preds.append(pred)
                    pbar.update(1)
                    pbar.refresh()

            # Calculating the mean across all TTA predictions
            mean_pred = torch.stack(tta_preds).mean(dim=0).cpu().numpy()
            all_preds.extend(mean_pred)

    return np.array(all_preds)


def run_inference(
    model_path: Path,
    output_path: Path,
    bands: np.ndarray,
    profile: Dict[str, Any],
    patch_size: int = 1000,
    overlap: int = 500,
    binary_output: bool = True,
    pbar: Optional[tqdm] = None,
    device: Optional[torch.device] = None,
) -> Path:
    if device is None:
        device = default_device()

    model = (
        pickle.load(
            open(model_path, "rb"),
        )
        .half()
        .to(device)
    )
    if pbar is None:
        pbar = tqdm(leave=False)

    pbar.reset()

    scene_size = bands.shape[-1]

    patches, locations = make_patches(bands, pbar, patch_size, overlap, scene_size)

    preds = inference(patches, model, device, batch_size=10, pbar=pbar)

    pred_array = stitch_preds(preds, locations, overlap, scene_size, pbar=pbar)

    export_pred(output_path, pred_array, profile, binary_output, pbar=pbar)
    # pbar.reset()
    pbar.refresh()
    pbar.set_description("Waiting to start Inference")
    pbar.n = 0
    pbar.total = 1
    del pred_array
    del preds
    gc.collect()

    return output_path
