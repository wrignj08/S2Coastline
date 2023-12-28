import rasterio as rio
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import gc
from torch import Tensor

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


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(band_stack, device):
    means = np.tile(MEANS[BAND_IDS], 6)
    stds = np.tile(STDS[BAND_IDS], 6)

    means_tensor = Tensor(means).view(1, -1, 1, 1).to(device).half()
    stds_tensor = Tensor(stds).view(1, -1, 1, 1).to(device).half()

    return (band_stack / 32767 - means_tensor) / stds_tensor


def create_gradient_mask(patch_size, patch_overlap_px):
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


def make_patches(band_stack, patch_size, overlap=20, scene_size=10980):
    patches, locations = [], []
    row_count = (scene_size - overlap) // (patch_size - overlap)
    b_bar = tqdm(total=row_count, desc="Making patches", leave=False)

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
        b_bar.update(1)

    b_bar.close()
    return patches, locations


def stitch_preds(preds, locations, overlap=20, scene_size=10980):
    gradient = create_gradient_mask(preds[0].shape[-1], overlap)
    pred_array = np.zeros((scene_size, scene_size))
    count_tracker = np.zeros((scene_size, scene_size))

    for pred, location in tqdm(
        zip(preds, locations), leave=False, desc="Stitching", total=len(preds)
    ):
        top, left = location
        pred_array[top : top + pred.shape[-1], left : left + pred.shape[-1]] = (
            pred_array[top : top + pred.shape[-1], left : left + pred.shape[-1]]
            + pred * gradient
        )
        count_tracker[
            top : top + pred.shape[-1], left : left + pred.shape[-1]
        ] += gradient
    pred_array = pred_array / count_tracker

    return pred_array


def export_pred(output_path, pred_array, profile, binary=True):
    profile["nodata"] = None
    if binary:
        profile.update(dtype=rio.int8, count=1, compress="lzw", driver="GTiff")
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(pred_array > 0, 1)
    else:
        profile.update(dtype=rio.float32, count=1, compress="lzw", driver="GTiff")
        with rio.open(output_path, "w", **profile) as dst:
            dst.write(pred_array, 1)


def batch_patches(patches, batch_size=10):
    batches = []
    for i in range(0, len(patches), batch_size):
        batches.append(np.array(patches[i : i + batch_size]))
    return batches


def inference(batches, model, device):
    model.eval()
    tta_depth = 6
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(batches, leave=False, desc="Inference"):
            batch = torch.tensor(batch.astype(np.float32)).to(device).half()
            batch = normalize(batch, device)

            tta_preds = []
            for tta in range(tta_depth):
                # Creating augmented batch for TTA
                augmented_batch = torch.roll(batch, shifts=2 * tta, dims=1)

                tta_preds.append(model(augmented_batch))

            # Calculating the mean across all TTA predictions
            mean_pred = torch.stack(tta_preds).mean(dim=0).cpu().numpy()
            all_preds.extend(mean_pred)

    return np.array(all_preds)


def run_inference(
    model_path,
    output_path,
    bands,
    profile,
    patch_size=1000,
    overlap=500,
    binary_output=True,
    device=None,
):
    if device is None:
        device = default_device()
    model_name = model_path.name

    model = (
        pickle.load(
            open(Path.cwd() / f"models/{model_name}", "rb"),
        )
        .half()
        .to(device)
    )
    model_path = Path.cwd() / f"models/{model_name}"

    scene_size = bands.shape[-1]

    patches, locations = make_patches(bands, patch_size, overlap, scene_size)

    del bands

    batches = batch_patches(patches, batch_size=10)
    del patches

    preds = inference(batches, model, device)

    pred_array = stitch_preds(preds, locations, overlap, scene_size)

    export_pred(output_path, pred_array, profile, binary_output)
    del pred_array
    del preds
    gc.collect()

    return output_path
