# %%
import rasterio as rio
import torch
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import pickle
import gc
from torch import Tensor
from threading import Thread
import time

# %%


# %%
means = [
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
stds = [
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
means = Tensor(means).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda().half()
stds = Tensor(stds).unsqueeze(1).unsqueeze(1).unsqueeze(0).cuda().half()


# %%
def normalise(band_stack):
    band_stack = band_stack / 32767
    band_stack = band_stack - means
    band_stack = band_stack / stds
    return band_stack


# %%
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


# %%
def make_patches(band_stack, patch_size, overlap=20, scene_size=10980):
    patches = []
    locations = []
    top = 0
    left = 0
    top_stop = False
    row_count = scene_size // (patch_size - overlap) + 1
    # print(row_count)
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


# %%
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


# %%
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


# %%
def inference(batches, model):
    preds = []
    with torch.no_grad():
        for batch in tqdm(batches, leave=False, desc="Inference"):
            batch = Tensor(batch.astype(np.float32)).cuda().half()
            batch = normalise(batch)
            pred = model(batch)
            pred = pred.cpu().detach().numpy()
            # un-batch preds
            for p in pred:
                preds.append(p)

            # preds.append(pred)
    return np.array(preds)


# %%
def run_inference(
    model_path,
    output_path,
    bands,
    profile,
    patch_size=1000,
    overlap=20,
    binary_output=True,
):
    # %%
    model_name = model_path.name

    # %%
    model = pickle.load(open(Path.cwd() / f"models/{model_name}", "rb")).half().cuda()
    model.eval()

    # %%

    # output_path = Path(str(raster_path).replace(".tif", "_pred.tif"))

    band_stack = bands

    scene_size = band_stack.shape[-1]

    patches, locations = make_patches(band_stack, patch_size, overlap, scene_size)

    del band_stack

    batches = batch_patches(patches, batch_size=10)
    del patches

    preds = inference(batches, model)

    pred_array = stitch_preds(preds, locations, overlap, scene_size)
    export_pred(output_path, pred_array, profile, binary_output)
    del pred_array
    del preds
    gc.collect()

    return output_path


# %%
