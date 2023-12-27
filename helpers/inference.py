import rasterio as rio
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import gc
from torch import Tensor

MEANS = [
    0.09561849,
    0.09644007,
    0.09435602,
    0.09631168,
    0.09356618,
    0.09504625,
    0.09509373,
    0.09508776,
    0.0911776,
    0.091464,
    0.09334985,
    0.09400712,
]
STDS = [
    0.02369863,
    0.03057647,
    0.0244495,
    0.03169953,
    0.02380443,
    0.03068336,
    0.02376207,
    0.03026029,
    0.02387124,
    0.03011121,
    0.02285621,
    0.02902071,
]


def default_device():
    """
    Determines the best available device for computation.

    This function checks if CUDA or MPS (Metal Performance Shaders) are
    available on the current machine, in that order. If neither are available,
    it defaults to using the CPU.

    Returns:
        torch.device: The device to be used for computation. This can be a CUDA
        device, MPS device or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalise(band_stack, device):
    means = Tensor(MEANS).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device).half()
    stds = Tensor(STDS).unsqueeze(1).unsqueeze(1).unsqueeze(0).to(device).half()
    band_stack = band_stack / 32767
    band_stack = band_stack - means
    band_stack = band_stack / stds
    return band_stack


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
    patches = []
    locations = []
    top = 0
    left = 0
    top_stop = False
    row_count = scene_size // (patch_size - overlap) + 1

    b_bar = tqdm(total=row_count, desc="Making patches", leave=False)
    while not top_stop:
        left_stop = False
        if top + patch_size > scene_size:
            top = scene_size - patch_size
            top_stop = True

        while not left_stop:
            if left + patch_size > scene_size:
                left = scene_size - patch_size
                left_stop = True
            patch = band_stack[:, top : top + patch_size, left : left + patch_size]

            patches.append(patch)

            locations.append((top, left))
            left += patch_size - overlap

        left = 0
        top += patch_size - overlap
        b_bar.update(1)
    # finish pbar
    b_bar.update(b_bar.total - b_bar.n)

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
    preds = []

    tta_depth = 6
    with torch.no_grad():
        for batch in tqdm(batches, leave=False, desc="Inference"):
            batch = Tensor(batch.astype(np.float32)).to(device).half()
            batch = normalise(batch, device)
            # make array to store tta preds, shape is (tta_depth, batch_size, classes, patch_size, patch_size)
            tta_preds = np.zeros(
                (tta_depth, batch.shape[0], 1, batch.shape[-2], batch.shape[-1])
            )
            # reorder the bands for each tta
            for tta in range(tta_depth):
                first_two = batch[:, :2, :, :]
                rest = batch[:, 2:, :, :]
                batch = torch.cat((rest, first_two), dim=1)

                pred = model(batch)
                # push to cpu and detach from graph
                pred = pred.cpu().detach().numpy()
                # store in tta_preds
                tta_preds[tta] = pred

            pred = np.mean(tta_preds, axis=0)

            for p in pred:
                preds.append(p)

    return np.array(preds)


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
        pickle.load(open(Path.cwd() / f"models/{model_name}", "rb")).half().to(device)
    )
    model.eval()

    band_stack = bands

    scene_size = band_stack.shape[-1]

    patches, locations = make_patches(band_stack, patch_size, overlap, scene_size)

    del band_stack

    batches = batch_patches(patches, batch_size=10)
    del patches

    preds = inference(batches, model, device)

    pred_array = stitch_preds(preds, locations, overlap, scene_size)
    export_pred(output_path, pred_array, profile, binary_output)
    del pred_array
    del preds
    gc.collect()

    return output_path
