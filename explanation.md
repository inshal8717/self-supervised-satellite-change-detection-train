# Repository Breakdown: `self-supervised-satellite-change-detection-train`

**Repository URL:** https://github.com/inshal8717/self-supervised-satellite-change-detection-train
**Package name:** `sat-change` v0.1.0

---

## Part 1 — What This Repository Is

An **end-to-end PyTorch pipeline** for **label-free satellite (Sentinel-2) change detection** built on three stages:

1. **Self-supervised pretraining (no labels):** A convolutional encoder is trained with **SimCLR-style temporal contrastive learning** using the **NT-Xent loss**. Two augmented views of the same image patch form a "positive pair"; all other patches in the batch are negatives. The encoder learns an invariant, discriminative representation of satellite imagery.
2. **Change detection (no fine-tuning, no labels):** The same pretrained encoder is used as a feature extractor. Before/after images of the same location pass through it, and the **cosine distance between dense multi-scale feature maps** yields a per-pixel "change score."
3. **Unsupervised thresholding:** Otsu's method (or a user threshold) converts the score map to a binary change mask.

### Key design facts

- **Not** trained as a Siamese pair network — it's single-image SimCLR pretraining; at inference the shared encoder is applied to both temporal images (Siamese *use*, not Siamese *training*).
- **No momentum encoder, no negative queue** — plain SimCLR with within-batch negatives only.
- **Fully convolutional small encoder** (3 downsampling stages) so **dense** per-pixel features can be extracted — unlike standard SimCLR which only needs a global vector.
- **Multi-scale comparison:** features from all 3 stages are L2-normalized, compared by cosine distance, upsampled, and averaged.
- **Satellite-safe augmentations:** flips, 90° rotations, per-band gain/bias jitter, Gaussian noise, occasional blur — no HSV color-space augs (multi-spectral data isn't RGB).
- **Robust normalization:** handles Sentinel-2 integer scale (0–10000) and float (0–1), clips to 2–98 percentiles, nodata → zeros.
- **Tiled inference with overlap blending** for large scenes on limited GPU memory.
- **Geographic-scene split recommended** to avoid spatial leakage.
- **No fine-tuning stage** — the method is pure feature-distance comparison.

### Architecture at a glance

```
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING (self-supervised)                    │
│                                                                  │
│  raw GeoTIFF/NumPy  →  read_image()  →  normalize_reflectance()  │
│       → random crop (patch) → two augmented views (augment())    │
│       → SatelliteEncoder → projection head → z1, z2              │
│       → NT-Xent (SimCLR) loss → AdamW + CosineAnnealingLR        │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                    INFERENCE (change detection)                  │
│                                                                  │
│  before.tif + after.tif  →  normalize → tile into patches        │
│       → SatelliteEncoder.features() (3 dense stages, 3 scales)   │
│       → cosine distance per scale → upsample → average           │
│       → dense change score map → Otsu → binary change mask       │
│       → optional IoU/F1/AUROC vs. ground truth mask              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 2 — File Structure

```
├── pyproject.toml            # build config; deps numpy/torch/Pillow; extras: geo(rasterio), test, dashboard
├── README.md                 # purpose, GEE export steps, methodology, limitations
├── app.py                    # Streamlit interactive dashboard (147 lines)
├── configs/sentinel2_example.json   # example config (B02,B03,B04,B08, 10m, temp 0.2...)
├── scripts/
│   ├── demo.py               # 1-command smoke demo (synthetic data → train → eval)
│   └── make_synthetic.py     # deterministic synthetic paired scenes with masks
├── src/sat_change/
│   ├── __init__.py           # package marker, __version__ = "0.1.0"
│   ├── io.py                 # read/write GeoTIFF & NumPy, normalization, scene keys, pairing (81 lines)
│   ├── data.py               # augment() + ContrastivePatchDataset (52 lines)
│   ├── model.py              # ConvBlock, SatelliteEncoder, nt_xent, feature_change (53 lines)
│   ├── train.py              # contrastive pretraining loop / CLI (53 lines)
│   ├── runtime.py            # load_model + tiled score_pair inference (41 lines)
│   ├── postprocess.py        # Otsu threshold + binary mask (23 lines)
│   ├── predict.py            # CLI: dense change map for one pair (27 lines)
│   ├── evaluate.py           # CLI: batch evaluation vs masks (32 lines)
│   └── metrics.py            # IoU/F1/precision/recall/accuracy + rank-based AUROC (28 lines)
└── tests/                    # test_io, test_metrics, test_model
```

---

## Part 3 — Line-by-Line Explanations

### 3.1 `pyproject.toml`

Packaging/build metadata for the `sat-change` package.

- **Lines 1–3:** Build system. Uses `setuptools>=68` with the `setuptools.build_meta` backend — the modern PEP 517 way to build the package (as opposed to legacy `setup.py`).
- **Lines 5–8:** Project identity: name `sat-change`, version `0.1.0`, description "Self-supervised satellite change detection with temporal contrastive learning", README as long description.
- **Line 10:** `requires-python = ">=3.10"` — Python version floor (needed for modern type syntax like `str | Path` used throughout the code).
- **Line 11:** Core runtime dependencies:
  - `numpy>=1.24` — array math for image IO and metrics.
  - `torch>=2.1` — the deep learning framework (autograd, `nn.Module`, optimizers, `DataLoader`).
  - `Pillow>=10` — image utilities (declared but not actually imported in the code; likely a transitive convenience).
- **Lines 13–16:** Optional extras:
  - `geo = ["rasterio>=1.3"]` — GeoTIFF support. This is **lazily imported** inside `io.py` functions so the core pipeline works without it.
  - `test = ["pytest>=7"]` — test runner.
  - `dashboard = ["streamlit>=1.38", "plotly>=5.22"]` — for `app.py`.
- **Lines 18–19:** `[tool.setuptools.packages.find] where = ["src"]` — **src-layout**: the package lives under `src/` so tests import `sat_change` without accidentally importing the working copy.
- **Lines 21–23:** Pytest config: `pythonpath = ["src"]` (make `sat_change` importable), `testpaths = ["tests"]`.

### 3.2 `src/sat_change/__init__.py`

Trivial package marker. Docstring plus `__version__ = "0.1.0"`. Nothing else — the heavy lifting lives in sibling modules.

### 3.3 `src/sat_change/io.py` (81 lines) — I/O, normalization, pairing

#### `read_image(path) -> (arr, profile)` — lines 10–27

- **Lines 13–17:** If the file suffix is `.npy`, load it with `np.load` and cast to `float32` (the canonical internal dtype). If it's 2-D (single band `H,W`), add a channel axis via `arr[None]` so everything downstream is consistently `(C,H,W)`. The geospatial `profile` is returned as an empty dict because a raw NumPy array has no CRS/transform metadata.
- **Lines 18–21:** For anything else (`.tif`/`.tiff`), lazily import `rasterio` (the `geo` extra). If missing, raise a helpful `RuntimeError` telling the user to `pip install -e '.[geo]'`. Lazy import keeps core dependencies minimal.
- **Lines 22–27:** Open with `rasterio.open(path)` in a context manager, read all bands with `src.read()` → shape `(C,H,W)`, cast to float32. `src.profile.copy()` captures geospatial metadata (CRS, affine transform, nodata, dtype, driver) — this profile is later reused to *write* GeoTIFF outputs in the correct georeferencing. If the source declares a nodata value, every pixel that equals nodata in *any* band is set to `np.nan` (line 26: `arr[:, np.any(arr == src.nodata, axis=0)] = np.nan`). This is a conservative "nodata mask" that propagates missingness across bands.

#### `write_image(path, arr, profile=None)` — lines 30–44

- **Lines 33–35:** `.npy` output → `np.save` as float32.
- **Lines 36–39:** GeoTIFF output requires rasterio; same friendly error if absent.
- **Lines 40–44:** If the array is 2-D, add a band axis. Build an output profile from the input profile (or empty dict), overriding driver=`GTiff`, height/width/count from the array, dtype=float32, and `compress="deflate"` (lossless LZ77-style compression that shrinks GeoTIFFs, especially masks). Write with `rasterio.open(path, "w", **out_profile)`. The original CRS/transform are preserved, so outputs overlay perfectly in GIS software.

#### `normalize_reflectance(arr)` — lines 47–60

This is the **preprocessing core**; ML-relevant details:

- **Lines 49–51:** Copy as float32. Compute `finite` mask (non-NaN, non-Inf). If the 99th percentile exceeds 2, the data is in **Sentinel-2 integer reflectance scale (0–10000)** (L2A values are often stored as DN = reflectance × 10000), so divide by 10000 to bring into 0–1. The `> 2` heuristic distinguishes "0..1 float reflectance" from "0..10000 integer" — reflectances are almost never above 2 in valid data, while the integer scale's 99th percentile almost always is.
- **Lines 53–59:** Per-band **robust min-max normalization**: for each band, take valid (finite) pixels, compute the **2nd and 98th percentiles** as `lo`/`hi` — this is *robust* because it ignores the extreme tails that would corrupt a plain min-max (sensor noise, specular reflections, clouds). Then `x = clip((x - lo) / max(hi - lo, 1e-6), 0, 1)`. The `1e-6` guard prevents division by zero on constant bands; clipping bounds outputs to [0,1], matching the augmentation assumptions and the encoder's input expectations. If a band has *no* valid pixels, it is zeroed (lines 55–57).
- **Line 60:** `np.nan_to_num(x)` converts any remaining NaN (nodata pixels) to 0. So nodata becomes zero reflectance — a deliberate, documented choice ("preserving nodata as zeros").

#### `scene_key(path)` — lines 63–64

Regex `[-_]?\d{4}[-]?\d{2}[-]?\d{2}$` strips a trailing date (`20230101`, `2023-01-01`, `2023_01_01`) from the filename stem, giving the **scene identifier**. This is the glue that groups temporal acquisitions of the same location.

#### `discover_images(root)` — lines 67–69

Recursively globs `root` for `.npy`/`.tif`/`.tiff` files, sorted for determinism.

#### `pair_images(root)` — lines 72–81

- **Lines 73–75:** Build a dict mapping `scene_key -> [paths...]` (grouping all dates of one scene).
- **Lines 76–80:** For scenes with ≥2 acquisitions, sort by filename (the date is part of the name, so sort ≈ chronological) and take **first vs last** as the temporal pair `(before, after)`. This gives the maximal temporal separation, which maximizes detectable change.

### 3.4 `src/sat_change/data.py` (52 lines) — augmentation & dataset

#### `augment(x)` — lines 14–28

This is the **data augmentation module** of SimCLR — it generates the two *views* that form a positive pair. The augmentations are deliberately "satellite-safe" (they don't distort spectral meaning):

- **Lines 16–17:** 50% chance horizontal flip (`x.flip(-1)`). Flips are valid because satellite imagery has no canonical orientation.
- **Lines 18–19:** 50% chance vertical flip (`x.flip(-2)`).
- **Line 20:** Random 90° rotation: `torch.rot90(x, k, dims=(-2,-1))` with k in {0,1,2,3}. Together with flips this gives the **dihedral group D4** of square symmetries. Only 90° rotations are used — arbitrary angles would create "empty corners" and break the rectangular patch assumption.
- **Lines 21–23:** **Spectral (per-channel) jitter**: `gain ~ Uniform(0.85, 1.15)` and `bias ~ Uniform(-0.05, 0.05)` broadcast over `(C,1,1)` — i.e., a different gain/bias per channel but spatially uniform. This models sun-angle, atmospheric and sensor-calibration differences *without* warping spatial layout. This mirrors SimCLR's "color jitter" but applied per-band instead of in RGB color space.
- **Lines 24–25:** 50% chance additive Gaussian noise, σ=0.015 — sensor noise robustness.
- **Lines 26–27:** 25% chance a 3×3 average pooling blur (`F.avg_pool2d(x[None], 3, 1, 1)[0]`) — models slight defocus/MTF differences; the `[None]` adds a batch dim and `[0]` removes it.
- **Line 28:** `x.clamp(0, 1)` ensures outputs stay in the valid reflectance range.

**ML concept — why augmentation matters in contrastive learning:** The whole premise of instance discrimination is that the two augmented views of the *same* image should map to nearby points in embedding space. Augmentations define the "equivariance/invariance prior": the encoder learns to ignore the augmentation-induced variation (flips, brightness, noise) while preserving the *identity* of the scene. If augmentations are too weak, the task is trivial; if too strong (e.g., cropping to 10% area, SimCLR's setting), the task is hard but features become more transferable. This repo keeps augmentations mild because the features must remain **spatially dense** (per-pixel) for change detection — destructive augmentations like aggressive cropping would destroy the dense correspondence.

#### `ContrastivePatchDataset` — lines 31–52

A `torch.utils.data.Dataset` that produces the positive pair `(view1, view2)`:

- **Lines 32–36 (`__init__`):** `discover_images(root)` finds all images; error if none. Stores `patch_size` (default 64), `samples_per_epoch` (default 4096), `bands` (default 4).
- **Lines 38–39 (`__len__`):** Returns `samples_per_epoch` rather than the number of images. This decouples epoch length from dataset size — you sample *random patches* from the archive until you've seen `samples_per_epoch` patches, giving control over epoch cost regardless of how many images exist.
- **Lines 41–51 (`__getitem__`):**
  - Line 42: `read_image(self.paths[index % len(self.paths)])` — **cyclic indexing**; every image is visited in turn as the epoch rolls, but the crop is random so each visit yields different patches.
  - Line 43: `normalize_reflectance(arr[:self.bands])` — band selection + normalization (see 3.3).
  - Lines 44–48: If the image is smaller than `patch_size`, pad with `np.pad(..., mode="reflect")` — **reflect padding** mirrors edge pixels, which is better for imagery than zero-padding (no artificial dark borders).
  - Lines 49–50: Random top-left corner `(y, x)` such that the patch fits.
  - Line 51: Extract the `(C, patch, patch)` patch and convert to a torch tensor (`np.copy()` ensures a writable contiguous buffer).
  - Line 52: **The contrastive pair:** `return augment(patch.clone()), augment(patch.clone())`. Two *independently* augmented views of the same patch. This is the positive pair; every other patch in the batch serves as a negative.

**ML concept — view generation:** In SimCLR, the pair `(x_i, x_j)` is the *same* image under two random augmentation draws. Note this dataset samples one patch per image per step; negatives come from other images/patches in the same batch. There is **no explicit temporal pairing during pretraining** — the "temporal" part of the README's "temporal contrastive pretraining" refers to the *downstream assumption* (the pretrained features are then compared across time), and to the fact that the archive contains multiple dates per scene (which makes the representation more robust). The pretraining itself is standard single-image SimCLR.

### 3.5 `src/sat_change/model.py` (53 lines) — the neural network core

#### `ConvBlock(nn.Sequential)` — lines 8–10

A residual-free double convolution block:

```
Conv2d(in→out, 3×3, stride, pad 1, bias=False) → BatchNorm2d → GELU
Conv2d(out→out, 3×3, stride 1, pad 1, bias=False) → BatchNorm2d → GELU
```

- **`bias=False` on the convs** because BatchNorm provides the bias term — avoids redundancy (standard practice).
- **BatchNorm2d** normalizes activations per-channel over the batch, which stabilizes training and accelerates convergence. In **inference** BatchNorm uses running statistics (the model must be in `.eval()` mode — note `feature_change` is decorated `@torch.no_grad()` and `score_pair` uses `torch.inference_mode()`, but BN statistics still need `eval()`; `load_model` calls `.eval()`).
- **GELU** (Gaussian Error Linear Unit) activation — a smooth, differentiable ReLU variant used in modern transformers; often slightly better gradient behavior than ReLU.
- The stride-1 second conv preserves spatial size; only the first conv (when `stride=2`) downsamples.

#### `SatelliteEncoder(nn.Module)` — lines 13–32

The **backbone** (feature extractor) plus **projection head**:

- **Line 15:** `__init__(in_channels=4, width=32, projection_dim=128)`.
- **Line 17:** `self.stem = ConvBlock(in_channels, width)` — first conv block, stride 1: `4 → 32` channels, spatial size preserved.
- **Lines 18–20:** Stages 1–3, each a `ConvBlock` with `stride=2` (downsampling by 2 each): `32 → 64 → 128 → 256` channels, spatial dims shrink by 8× total (a 64×64 patch → 8×8 at the deepest stage).
- **Line 21:** **Projection head** `self.projector = Linear(256, 256) → GELU → Linear(256, 128)`. In SimCLR, the projection head is an MLP that maps representations `h` to a lower-dimensional embedding `z` where the contrastive loss is computed. The head discards task-irrelevant information (e.g., exact color statistics) during pretraining, and after pretraining it is *discarded* and only the backbone `h` is used — which is exactly what happens here: `feature_change` uses `model.features()` (the backbone), never `model.projector`.
- **Lines 23–28 (`features`):** Forward through stem → stages 1–3, returning `[b, c, d]` — the **three dense feature maps** at 1/2, 1/4 and 1/8 resolution. This is the "dense feature stages" design: unlike a typical classification backbone that only needs the final global pooled vector, this encoder keeps intermediate feature maps so change detection can be done per-pixel at multiple scales.
- **Lines 30–32 (`forward`):** The pretraining path. `z = self.features(x)[-1].mean((-2,-1))` — global average pooling (GAP) of the deepest feature map produces a 256-dim vector. `F.normalize(self.projector(z), dim=1)` projects to 128 dims and **L2-normalizes** to the unit hypersphere. Normalization is important for the NT-Xent loss because the loss is a function of *cosine similarity*, and L2-normalization makes cosine similarity equal to the dot product.

**ML concept — cosine similarity & the unit hypersphere:** After L2 normalization, every embedding lies on the surface of a 128-dimensional unit sphere. The dot product of two such vectors equals the cosine of the angle between them. This is the similarity measure used by SimCLR: positives should be close (dot product → 1, angle → 0°), negatives far apart.

#### `nt_xent(z1, z2, temperature=0.2)` — lines 35–42

The **NT-Xent** (Normalized Temperature-scaled Cross-Entropy) loss, the SimCLR loss. This is the entire training objective.

- **Line 37:** `n = z1.shape[0]` — batch size.
- **Line 38:** Concatenate `[z1, z2]` along the batch dim → shape `(2n, 128)`, then L2-normalize. All 2n views form the comparison pool.
- **Line 39:** `logits = z @ z.T / temperature` — the full `(2n, 2n)` matrix of pairwise **cosine similarities scaled by 1/T**. The temperature T (default 0.2) sharpens the softmax: small T makes the distribution peakier (harder negatives get more gradient pressure); large T makes it flatter/softer. T is a critical hyperparameter — too small causes training collapse via overly aggressive gradients; too large makes the loss uninformative.
- **Line 40:** `logits.fill_diagonal_(-torch.inf)` — zero out the *self-similarity* diagonal (each view compared with itself = similarity 1). Setting to `-inf` means it contributes nothing to the softmax.
- **Line 41:** `targets = torch.cat([arange(n, 2n), arange(n)])` — the label assignment: for view `i` in the first half, the positive is view `i+n` (its augmented twin in the second half); for view `i+n` in the second half, the positive is view `i`. So the positive of `z1[i]` is exactly `z2[i]` — a symmetric pairing.
- **Line 42:** `F.cross_entropy(logits, targets)` — for each of the 2n rows, a softmax over the 2n−1 other views, with the correct positive as the target class. The loss for view i is `-log( exp(sim(z_i, z_pos)/T) / Σ_j exp(sim(z_i, z_j)/T) )`, averaged over all 2n rows.

**ML concept — why this is "instance discrimination":** Each image patch is treated as its own class. The model must pull the two augmented views together while pushing every other patch in the batch apart. With batch size 64, each row sees 126 negatives — the model cannot cheat by clustering (which would be collapse) because every patch must be distinguishable from every other patch. This "uniformity" pressure on the hypersphere is what makes SimCLR features transferable.

**ML concept — temperature and numerical stability:** `logits / T` with normalized vectors is bounded in magnitude (~±1/0.2 = ±5), so softmax is numerically safe. `fill_diagonal_(-inf)` and `cross_entropy` handle the log-sum-exp stably.

#### `feature_change(model, before, after)` — lines 45–52

The **inference-time change detector** (a `@torch.no_grad()` function):

- **Line 47:** `target_size = before.shape[-2:]` — we will resample every scale back to the original input resolution.
- **Line 49:** For each of the three feature stages, zip `model.features(before)` with `model.features(after)` — the encoder runs once on each image, giving dense feature maps `f1` (before) and `f2` (after) at that scale.
- **Line 50:** L2-normalize both feature maps along the channel dim (`dim=1`) — per-pixel unit-normalized feature vectors.
- **Line 51:** `distance = 1 - (f1 * f2).sum(1, keepdim=True)` — **cosine distance** per pixel: the dot product of normalized vectors is the cosine similarity; `1 − cos` is the distance. Where features match (unchanged land), similarity ≈ 1 → distance ≈ 0; where they diverge (change), distance is large. `keepdim=True` preserves a channel dim so `F.interpolate` works.
- **Line 52:** `F.interpolate(distance, target_size, mode="bilinear", align_corners=False)` — upsample the coarse-scale distance maps back to full resolution. Bilinear interpolation smooths boundaries appropriately.
- **Line 53:** `torch.stack(scores).mean(0).squeeze(1)` — **average the three scales** → final dense change score map `(H, W)`. Averaging multi-scale distances fuses fine-grained detail (stage 1) with robust semantic context (stage 3).

**ML concept — why compare dense features instead of fine-tuning a classifier:** After pretraining, the encoder has learned that "the same location looks the same at different times unless the world changed." This makes the *feature space itself* the change detector: no labeled data, no fine-tuning, no classification head. This is the core trick of the whole repository. Note that this is **not a Siamese network in the training sense** — it's a *shared-weight* feature extractor applied to two inputs at inference time (the weights are shared by definition since it's the same model, which is what a Siamese network is), but there was no contrastive *pair* training on before/after data.

### 3.6 `src/sat_change/train.py` (53 lines) — pretraining loop

#### `parse_args()` — lines 16–25

CLI surface with defaults matching `configs/sentinel2_example.json`:
- `--data`, `--output` (required).
- `--epochs 50`, `--batch-size 64`, `--patch-size 64`, `--samples-per-epoch 4096`.
- `--bands 4`, `--width 32` (base channel count), `--projection-dim 128`, `--temperature 0.2`, `--lr 3e-4`, `--workers 0`, `--seed 42`.
- `--device` auto-selects CUDA if available, else CPU.

#### `main()` — lines 28–49

- **Line 29:** Seeds `random`, `np.random`, `torch.manual_seed` for reproducibility (torch's CUDA RNG is also seeded via `torch.manual_seed` in modern versions).
- **Line 30:** Creates the output directory.
- **Line 31:** Builds `ContrastivePatchDataset`.
- **Line 32:** `DataLoader(ds, batch_size, shuffle=True, num_workers, drop_last=True, pin_memory=cuda)`. `shuffle=True` randomizes patch-image order each epoch; `drop_last=True` guarantees the final batch has exactly `batch_size` samples — **critical for NT-Xent** because the loss symmetrically pairs `z1[i] ↔ z2[i]` within one batch (an odd-sized final batch would break the pairing arithmetic); `pin_memory` speeds host→GPU copies.
- **Line 33:** Instantiate `SatelliteEncoder` on the device.
- **Line 34:** `AdamW(lr=3e-4, weight_decay=1e-4)` — AdamW is Adam with **decoupled weight decay** (weight decay is applied directly to weights, not through the gradient like L2), which gives better generalization than classic Adam. 1e-4 is a mild regularizer.
- **Line 35:** `CosineAnnealingLR(optimizer, epochs)` — the LR follows a cosine curve from `lr` down to ~0 over the full run. Cosine schedules give smooth convergence; the monotonic decay matters because there's no validation-based checkpoint selection beyond lowest loss.
- **Line 36:** `config = vars(args)` (saved into checkpoints for reproducibility), `best = inf` (track lowest loss), `history = []` (loss curve for plotting).
- **Lines 37–44 (the epoch loop):**
  - Line 38: `model.train()` — enables dropout/BatchNorm training behavior; `total` accumulates the loss.
  - Lines 39–43 (the batch loop): move views to device; `optimizer.zero_grad(set_to_none=True)` (faster than zeroing to zeros); compute `nt_xent(model(x1), model(x2), temperature)` — note the **global average pooled + projected** embeddings from `forward()`; `loss.backward()` computes gradients; `optimizer.step()` applies them; accumulate loss.
  - Line 44: `scheduler.step()` after each epoch; average loss for the epoch; append to history.
- **Lines 45–47:** Save `state = {model, optimizer, epoch, loss, config}` as `last.pt` every epoch and as `best.pt` whenever the epoch loss is a new minimum. Keeping the optimizer state enables training resumption; keeping `config` lets `load_model` reconstruct the exact architecture (bands/width/projection_dim).
- **Line 49:** Write `history.json` — the loss curve.

**ML concept — no labels, no validation:** The training signal is entirely self-generated from data structure. Because there is no labeled validation set, model selection uses the lowest pretraining loss — acceptable because the representation quality correlates with lower contrastive loss (lower loss ≈ better-formed hypersphere embeddings).

### 3.7 `src/sat_change/runtime.py` (41 lines) — inference plumbing

#### `load_model(checkpoint, device="cpu")` — lines 12–17

- **Line 13:** `torch.load(checkpoint, map_location=device, weights_only=False)` — loads the state dict dict; `map_location` allows loading a CUDA-trained checkpoint on CPU or vice versa; `weights_only=False` allows deserializing the plain dict (needed because the checkpoint contains config/optimizer state; in newer PyTorch versions weights_only defaults to True for safety).
- **Line 14:** `config = payload.get("config", {})` — recover hyperparameters saved at training time.
- **Lines 15–17:** Rebuild `SatelliteEncoder(bands, width, projection_dim)` from config (with fallbacks 4/32/128), `load_state_dict`, move to device, `.eval()`. Eval mode is essential for correct BatchNorm statistics and any dropout behavior.

#### `score_pair(model, before_path, after_path, device, tile_size=512, overlap=32)` — lines 20–41

The **tiled dense change-scoring engine**:

- **Line 21:** Read both images (arr + profile from the before image).
- **Lines 22–23:** Shape check — before/after must be identical `(C,H,W)`. This enforces **pixel-level coregistration** up front; the README explicitly warns the user to register (align) acquisitions in GEE before export. This is a *hard* domain constraint: change detection on misaligned images is meaningless.
- **Line 24:** `bands = model.stem[0].in_channels` — read the input channel count *from the model* (not from the image), so any checkpoint works regardless of the source band count.
- **Lines 25–26:** `normalize_reflectance(before[:bands])` and same for after → torch tensors.
- **Lines 27–29:** Validate tile geometry: `tile_size > overlap` and `tile_size >= 16` (the smallest the 8× downsampled encoder can represent meaningfully: 16/8 = 2 px at the deepest scale).
- **Line 30:** Accumulators: `total` (sum of tile scores) and `weights` (tile overlap count) — both `(H,W)` float32. These implement **overlap blending** (see below).
- **Lines 31–33:** Build the tile grid. Y positions step by `tile_size - overlap` so adjacent tiles overlap by `overlap` pixels; the same for X. Then ensure the final tile always covers the image's bottom/right edge (`ys[-1] == max(0, h - tile_size)`); if a tile starts before the edge but would end past it, we don't slide the grid — instead we add an explicit final tile anchored at the edge. This guarantees **full image coverage**.
- **Lines 34–40:** `torch.inference_mode()` (faster than `no_grad` — skips autograd bookkeeping entirely). For each tile: slice the pre-normalized arrays `a[:, y:y+ts, x:x+ts][None]` (add batch dim), move to device, run `feature_change(model, aa, bb)` → `(1, H_t, W_t)` score, back to CPU/NumPy. Accumulate `total[y:y+th, x:x+tw] += score` and `weights[...] += 1` in the *original image coordinates*.
- **Line 41:** `total / maximum(weights, 1)` — **weighted average blending**: overlapping regions get averaged over the number of tiles covering them, which removes tile-boundary seams (the "overlap blending" of the README). Return `(score_map, profile)`.

**Why tiling?** Memory: a large scene (e.g., 10k×10k pixels) cannot be run through the encoder at once on typical GPUs. Tiling with overlap also avoids hard tile-edge artifacts, since each pixel's score is a blend of multiple overlapping passes. The cost is a small constant factor of redundant compute in the overlap zones.

### 3.8 `src/sat_change/postprocess.py` (23 lines) — thresholding

#### `otsu_threshold(values, bins=256)` — lines 6–17

**Otsu's method** is the classic unsupervised thresholding algorithm: it finds the threshold that **maximizes between-class variance** (equivalently minimizes within-class variance), assuming the histogram is roughly bimodal (two classes: changed / unchanged pixels).

- **Lines 7–9:** Use only finite values, cast to float64. If empty or constant (`np.ptp(x) == 0`), return the value itself (or 0.0) — no meaningful split exists.
- **Line 10:** Histogram with 256 bins; `edges`/`centers` give bin boundaries/midpoints.
- **Lines 12–13:** `weight1 = cumsum(hist)` — cumulative weight of bins 0..k (class "below"); `weight2 = cumsum(hist[::-1])[::-1]` — cumulative weight of bins k..255 (class "above"). These are the *probabilities* of each class for every candidate split.
- **Lines 14–15:** `mean1 = cumsum(hist * centers)/max(weight1,1)` — cumulative mean of class "below" at each split; `mean2` analogously for "above" (computed by reversing).
- **Line 16:** `variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2` — the **between-class variance** for each split point k: `σ²_B(k) = w1(k)·w2(k)·(μ1(k) − μ2(k))²`. (The standard Otsu formula also has a factor of the total variance in the denominator, but since the total variance is constant across splits, maximizing this expression is equivalent.)
- **Line 17:** Return the bin center at the split that maximizes between-class variance — the Otsu threshold.

#### `binary_mask(score, threshold=None)` — lines 20–22

- **Line 21:** If no threshold given, compute Otsu; else use the provided one.
- **Line 22:** `(score >= threshold)` → uint8 mask `{0,1}`; return `(mask, threshold)`. The threshold is returned so callers can log/report it.

**ML concept — label-free thresholding:** Since change detection here is unsupervised, the binary decision must also be unsupervised. Otsu's assumption — the score histogram is bimodal with a valley between "no change" and "change" — is reasonable when changes are localized but imperfect when change covers a large fraction of the scene (the two classes merge). This is a documented limitation; the README suggests fitting thresholds on validation regions for honest evaluation.

### 3.9 `src/sat_change/predict.py` (27 lines) — single-pair CLI

- **Lines 16–18:** CLI: `--before`, `--after`, `--checkpoint`, `--output` (required); `--threshold` (optional float; absent → Otsu), `--tile-size 512`, `--overlap 32`, `--device`.
- **Line 19:** Load model, run `score_pair` → dense score map + profile.
- **Line 20:** `binary_mask(score, threshold)` → mask + chosen threshold.
- **Lines 21–22:** Write the float score map (e.g., `scene000_change.npy` / `.tif` with the original geospatial profile) and the mask at `output_stem + "_mask" + suffix` (e.g., `scene000_change_mask.npy`).
- **Line 23:** Write `output.with_suffix(".json")` metadata: `{"threshold": ..., "changed_fraction": mask.mean()}` — the fraction of pixels flagged as changed.
- **Line 24:** Console summary.

### 3.10 `src/sat_change/evaluate.py` (32 lines) — batch evaluation CLI

- **Lines 17–18:** CLI: `--images`, `--masks`, `--checkpoint`, `--output` (required); optional `--threshold`, `--tile-size`, `--overlap`, `--device`.
- **Line 19:** Load the model once.
- **Line 20:** Index masks by `scene_key(filename)` so they can be matched to image pairs.
- **Lines 21–26 (the per-scene loop):**
  - Line 21: `pair_images(images)` yields `(before, after)` for every scene with ≥2 acquisitions.
  - Line 22–23: Skip scenes without a mask.
  - Line 24: Score the pair and threshold (Otsu if `--threshold` absent).
  - Line 25: `target, _ = read_image(masks[key]); target = target[0] > 0.5` — the mask is read as `(C,H,W)`; take band 0 and binarize at 0.5 (handles both float probability masks and uint8 0/1 masks).
  - Line 26: Build the per-scene result row: scene name, threshold, all segmentation metrics, plus `rank_auc`.
- **Line 27:** Error if no pairs matched masks (typo in mask names, etc.).
- **Line 28:** `summary = {k: nanmean over scenes}` — **NaN-mean** over scenes so a scene with a NaN AUROC (e.g., all-positive or all-negative target) doesn't poison the average. Only numeric metric keys are averaged (scene/threshold excluded).
- **Line 29:** Write `{summary, scenes}` to JSON and print the summary.

### 3.11 `src/sat_change/metrics.py` (28 lines)

#### `segmentation_metrics(pred, target)` — lines 6–16

Standard confusion-matrix statistics for binary masks:
- **Line 7:** Flatten and cast both to bool.
- **Line 8:** `tp = pred & target`, `fp = pred & ~target`, `fn = ~pred & target`, `tn = ~pred & ~target`, all summed.
- **Lines 9–10:** `precision = TP/(TP+FP)` (of predicted changes, how many are real), `recall = TP/(TP+FN)` (of real changes, how many found).
- **Lines 11–15:** `iou = TP/(TP+FP+FN)` (**Intersection over Union** / Jaccard — the standard change-detection metric), `f1 = 2·P·R/(P+R)` (harmonic mean of precision/recall), accuracy. All denominators are guarded with `max(..., 1)` / `1e-12` to avoid division by zero on degenerate inputs.
- **No `average precision` despite the README claiming it** — AUROC is implemented instead (README's "average precision" is aspirational/oversold).

**ML concept — why precision/recall matter for change detection:** Change pixels are rare (class imbalance). Accuracy is misleading (predicting "no change" everywhere gives high accuracy). Precision/recall and F1/IoU directly measure the useful behavior; AUROC measures ranking quality independent of threshold.

#### `rank_auc(score, target)` — lines 19–27

**Dependency-free AUROC** computed from pairwise rank statistics (the Mann–Whitney U statistic), avoiding sklearn:
- **Lines 21–22:** Flatten; split scores by positive/negative labels.
- **Lines 23–24:** If either class is empty, return NaN (can't rank).
- **Lines 25–26:** `order = argsort(s, kind="stable")`; then `ranks[order] = arange(1, n+1)` — compute each pixel's rank (1 = lowest score) via inverse permutation. Stable sort breaks score ties by index deterministically.
- **Line 27:** `AUROC = (Σ ranks_of_positives − n_pos·(n_pos+1)/2) / (n_pos·n_neg)`. This is the Mann–Whitney U statistic normalized: it equals the fraction of (positive, negative) pairs where the positive scores higher. 1.0 = perfect separation, 0.5 = random.

### 3.12 `scripts/make_synthetic.py` (43 lines) — synthetic data generator

Produces deterministic paired scenes so the whole pipeline can be smoke-tested without downloading real Sentinel-2 data.

#### `smooth(x, rounds=5)` — lines 9–12
A cheap box-blur: each pixel becomes the average of itself and its 4 neighbors (`np.roll` shifts the array in 4 directions). Repeated rounds → smooth, spatially-correlated random field (mimics natural texture coherence of land cover).

#### `make_scene(rng, size, kind)` — lines 15–29
- **Line 16:** `field = smooth(rng.random((size,size)), 10)` — smooth base texture.
- **Line 17:** Build a 4-band "before" image (band semantics ≈ B02,B03,B04,B08: blue, green, red, NIR) with per-band linear transforms of the field: `[.12+.15f, .2+.3f, .12+.12f, .35+.5f]` — different base offsets/gains give each band distinct brightness/contrast. The NIR band (index 3) has the strongest dynamic range — realistic for vegetation.
- **Lines 18–21:** `after = before.copy()`; make an elliptical change region: `((yy−cy)/ry)² + ((xx−cx)/rx)² ≤ 1` — a filled ellipse with random center/radii. This is the ground-truth mask.
- **Lines 22–27:** Apply one of **three change types** (rotated by `kind % 3`):
  - `deforestation` (kind%3==0): vegetation→soil, after values `[.32, .28, .22, .25]` — NIR drops from 0.35–0.85 to 0.25 (vegetation has high NIR reflectance; soil much lower — this is the spectral signature of vegetation loss).
  - `urbanization` (kind%3==1): bright built surface `[.5, .48, .5, .25]` — high visible, low NIR.
  - `flooding` (kind%3==2): dark water `[.08, .09, .07, .03]` — very low everywhere, especially NIR.
- **Lines 28–29:** Add Gaussian sensor noise σ=0.01 to both, clip to [0,1].

#### `main()` — lines 32–39
- **Lines 33–34:** CLI (`--output`, `--scenes 12`, `--size 128`, `--seed 7`); create `images/` and `masks/`.
- **Lines 36–38:** For scene i: save `scene{NNN}_20230101.npy` (before), `scene{NNN}_20240101.npy` (after), and mask `scene{NNN}.npy` — exactly the filename convention `scene_key()` parses.

### 3.13 `scripts/demo.py` (14 lines)

One-command smoke test. `run()` (lines 7–8) invokes the current Python interpreter as a subprocess (`sys.executable`) with `check=True` so any failure aborts the demo.
- **Line 11:** generate 8 synthetic scenes of size 96.
- **Line 12:** pretrain for just **2 epochs**, 128 samples/epoch, batch 16, patch 48 — tiny, CPU-friendly.
- **Line 13:** evaluate on all pairs with the trained `best.pt` → `outputs/demo/metrics.json`.

### 3.14 `app.py` (147 lines) — Streamlit dashboard

- **Lines 1–14:** Imports; `from __future__ import annotations` for forward refs.
- **Line 17:** Page config: title, satellite emoji favicon, wide layout.
- **Lines 20–24 (`_display_image`):** Convert a `(C,H,W)` array to a displayable RGB `(H,W,3)` image: take first 3 bands (or replicate band 0), move channels last, then **2–98 percentile stretch** (clip contrast to the data's informative range) and normalize to [0,1].
- **Lines 27–32 (`_load_uploaded`):** Save an uploaded file to a temp file with its original suffix (so `read_image` can infer format), returning the path.
- **Lines 35–38 (`_npy_bytes`):** Serialize an array to in-memory `.npy` bytes for download.
- **Lines 41–54 (`_geotiff_bytes`):** If a geospatial profile exists, write the array to an in-memory GeoTIFF (single-band, float32, deflate) using `rasterio.io.MemoryFile`; returns bytes or None (no rasterio/profile → the caller falls back to `.npy`).
- **Lines 57–71 (`_render_map`):** If CRS/transform exist, extract changed-pixel coordinates (`np.where(mask > 0)`), **subsample to ≤5000 points** for performance, map pixel→geo coords with `rasterio.transform.xy`, reproject to EPSG:4326 lon/lat, and draw red dots on `st.map`. Any import/geometry failure degrades to a warning.
- **Lines 74–147 (`main`):**
  - Lines 78–88: Sidebar: checkpoint picker (glob `outputs/**/*.pt`), file uploaders for before/after/mask, tile size (512), overlap (32), threshold (0 = Otsu), Run button.
  - Lines 90–95: Early returns when not run / no checkpoint.
  - Lines 97–102: Save uploads to temp files, then `score_pair(...)` under a spinner.
  - Line 102: `binary_mask(score, None if threshold_value==0 else threshold_value)` — threshold override or Otsu.
  - Lines 104–108: Re-read inputs for display; **delete temp files**; `changed_fraction = mask.mean()`.
  - Lines 110–114: Four metric cards: threshold, changed-area %, image size, changed-pixel count.
  - Lines 116–125: If a ground-truth mask was uploaded, compute `segmentation_metrics(mask, target[0] > 0.5)` and `rank_auc`, show five metric cards.
  - Lines 127–133: Results: plotly `Heatmap` of the score (Turbo colorscale), before/after images, binary mask.
  - Lines 135–136: Geospatial layer.
  - Lines 138–143: Download buttons: metadata JSON, score, and mask (GeoTIFF if georeferenced else `.npy`).

### 3.15 `configs/sentinel2_example.json`

A documented example config matching the CLI defaults:
- `bands`: `["B02","B03","B04","B08"]` — Sentinel-2 L2A **10 m** resolution bands (blue, green, red, NIR).
- `resolution_m: 10` — the grid to resample everything to.
- `cloud_probability_max: 40` — cloud mask threshold for GEE export.
- `patch_size: 64`, `batch_size: 64`, `epochs: 50`, `learning_rate: 0.0003`, `temperature: 0.2`.
- `split_strategy: "geographic_scene"` — evaluation split by scene to avoid spatial leakage (patches from the same scene in both train and test would inflate scores).

### 3.16 Tests (54 lines total)

- **`tests/test_io.py`:** (a) `scene_key("tile_a_2023-01-01.tif") == "tile_a"` and that only the scene with two dates forms a pair; (b) normalization of a 0..10000 array containing a NaN yields finite values in [0,1] (validates the integer-scale heuristic and nodata handling).
- **`tests/test_metrics.py`:** (a) perfect prediction → F1=1, AUROC=1; (b) a bimodal score map (100 zeros + 100 ones) → Otsu separates exactly 100 pixels.
- **`tests/test_model.py`:** `pytest.importorskip("torch")` skips if torch missing. (a) With `width=8`, `projection_dim=16`, random batch: `nt_xent` returns a finite positive loss. (b) `feature_change` on a single pair returns shape `(1, 32, 32)` with non-negative scores (cosine distance ≥ 0 after normalization — validates the `1 − cos` formula's bounds).

---

## Part 4 — The Data Flow (End to End)

### 4.1 Pretraining data flow

```
Sentinel-2 GeoTIFFs (B02,B03,B04,B08, 0..10000 int or 0..1 float)
  → read_image: (C,H,W) float32, nodata → NaN            [io.py:10]
  → normalize_reflectance: /10000 if needed; per-band 2–98 pct
      min-max to [0,1]; NaN → 0                           [io.py:47]
  → ContrastivePatchDataset.__getitem__: pick image by index,
      random reflect-padded crop (64×64)                  [data.py:41]
  → augment() twice independently                         [data.py:14]
      (flip H, flip V, rot90, per-band gain/bias, noise, 3×3 blur, clamp)
  → batch of (view1, view2) pairs via DataLoader          [train.py:32]
  → SatelliteEncoder.forward: stem→stage1→stage2→stage3,
      GAP of deepest map → projection head → L2-normalized z  [model.py:30]
  → NT-Xent loss across the 2n views (temp 0.2)           [model.py:35]
  → AdamW + CosineAnnealingLR update                      [train.py:34–35]
  → best.pt / last.pt / history.json                      [train.py:45–49]
```

### 4.2 Change detection data flow

```
before.tif + after.tif (registered, same CRS/transform/shape)
  → read_image both
  → shape check (coregistration enforcement)              [runtime.py:22]
  → normalize_reflectance (bands from model.stem[0].in_channels)
  → tile grid (512×512, 32 overlap, edge-anchored)        [runtime.py:31]
  → per tile: SatelliteEncoder.features(before) & (after)
      → 3 dense stages × 2 images
      → per-stage: L2 normalize channels → 1 − dot product
          (per-pixel cosine distance)                      [model.py:49–51]
      → bilinear upsample to full res → average 3 scales  [model.py:52–53]
  → overlap-weighted blending of tiles → dense score map  [runtime.py:40]
  → Otsu (or user threshold) → binary change mask         [postprocess.py]
  → optional: IoU/F1/precision/recall/AUROC vs ground truth [metrics.py]
  → outputs: score .tif/.npy, mask .tif/.npy, metadata.json
```

---

## Part 5 — Key Hyperparameters and Their Roles

| Hyperparameter | Default | Location | Role |
|---|---|---|---|
| `temperature` (τ) | 0.2 | train.py:22, model.py:35 | NT-Xent sharpness. Scales logits before softmax. Lower = harder contrastive task, better separation, risk of instability. |
| `batch_size` | 64 | train.py:19 | Defines the number of negatives (2n−1 per view). Larger batch = more negatives = better SimCLR features (SimCLR scales with batch size). |
| `patch_size` | 64 | train.py:20, data.py:32 | Pretraining crop size. After 8× downsampling, deepest features are 8×8. Must be ≥16 for meaningful dense features. |
| `samples_per_epoch` | 4096 | train.py:20 | Number of random patches per epoch (decouples epoch cost from archive size). |
| `width` | 32 | train.py:21 | Base channel count; stages become 32/64/128/256. Capacity knob. |
| `projection_dim` | 128 | train.py:22 | Embedding dimension where NT-Xent is computed. |
| `lr` | 3e-4 | train.py:23 | AdamW learning rate (standard for Adam-family). |
| `weight_decay` | 1e-4 | train.py:34 | AdamW decoupled L2 regularization. |
| scheduler | CosineAnnealingLR | train.py:35 | LR decays cosinely to ~0 over `epochs`. |
| `epochs` | 50 | train.py:19 | Total pretraining epochs. |
| `seed` | 42 | train.py:24 | Reproducibility. |
| `tile_size` / `overlap` | 512 / 32 | predict.py:18, runtime.py:20 | Inference memory/quality trade-off; overlap blending removes seams. |
| `threshold` | Otsu | predict.py:17, postprocess.py:20 | Binarization; user can override with fixed value. |
| augmentation probs | 0.5/0.5/0.25 | data.py:16–27 | Flip/noise/blur probabilities; mild by design. |
| `gain`/`bias` ranges | [0.85,1.15]/[−0.05,0.05] | data.py:21–22 | Spectral jitter strength. |

---

## Part 6 — Concepts Glossary (for teaching)

- **Self-supervised learning:** Training on unlabeled data by constructing a surrogate (pretext) task from the data itself. Here: instance discrimination via NT-Xent.
- **Contrastive learning:** Learn embeddings where positive pairs are close and negative pairs are far apart. Here, positives = two augmented views of one patch; negatives = all other patches in the batch.
- **SimCLR:** The specific framework — (1) random augmentation → 2 views, (2) shared encoder, (3) projection head, (4) NT-Xent loss, (5) discard the projection head after pretraining. This repo implements exactly this.
- **NT-Xent (Normalized Temperature-scaled Cross-Entropy):** Softmax cross-entropy over the 2n−1 other views, treating the correct positive as the class; temperature-scaled.
- **Siamese network:** A network applied to two inputs with **shared weights**. In this repo the same `SatelliteEncoder` processes before and after images at inference (so the scoring is Siamese), but *training* is not Siamese-pair training — it's single-image SimCLR. The README's "temporal contrastive pretraining" is thus temporal only in downstream use.
- **Momentum encoder:** A slowly-updated (EMA) copy of the encoder used in BYOL/MoCo to generate consistent target representations. **Not used here** — noted because it is a common contrastive component the reader might expect; this codebase deliberately stays with simple SimCLR.
- **Projection head:** MLP (Linear→GELU→Linear) mapping backbone features `h` to the loss space `z`; discarded after pretraining because it absorbs augmentation-specific information.
- **Backbone:** The convolutional feature extractor (stem + 3 stages) — the part that survives to inference.
- **Dense features:** Feature maps that retain spatial layout (vs. a single global vector), enabling per-pixel change scoring.
- **Augmentation / view generation:** Random transforms creating two distinct but semantically identical versions of an image; defines the invariance prior.
- **Cosine distance / similarity:** `1 − cos(θ)` between per-pixel normalized feature vectors; the change score.
- **GAP (global average pooling):** Mean over spatial dims → vector; used only in the pretraining head.
- **Otsu's method:** Threshold maximizing between-class variance; unsupervised binarization.
- **AUROC / IoU / F1 / precision / recall:** Threshold-independent ranking quality / overlap / harmonic-mean metrics.
- **Spatial leakage:** Training and testing on patches from the same scene overstates performance; the README mandates scene-level splits.
- **Coregistration:** Pixel-alignment of multi-date imagery; enforced via shape equality in `score_pair`.
- **Robust normalization:** 2–98 percentile clipping, immune to outliers — suitable for sensor noise/clouds.

---

## Part 7 — Limitations & Honest Assessment

1. **No registration or cloud masking in the pipeline** — inputs must be pre-aligned and cloud-free (GEE workflow in README).
2. **Small local archives may not transfer across biomes** — contrastive features are data-dependent.
3. **Otsu assumption** (bimodal score histogram) breaks when change area is large.
4. **No momentum encoder/queue** — SimCLR with batch-size-limited negatives; smaller effective negative counts than SOTA.
5. **No fine-tuning stage for change detection** — the method is pure feature-distance; no learned change classifier.
6. **AUROC is implemented, but the README claims "average precision"** — not present in `metrics.py`.
7. **`drop_last=True`** discards the last partial batch (minor data loss) — necessary for NT-Xent pairing.
8. **Model selection by lowest pretraining loss only** — no labeled validation signal.
9. **BatchNorm + small batches** can be unstable; `batch-size 16` in the demo is far from ideal.

---

## Part 8 — Verdict

This repository is best described as a **clean, minimal, teaching-quality reference implementation** of "SimCLR-pretrained dense feature comparison for unsupervised satellite change detection," prioritizing simplicity and reproducibility (synthetic data + tests + demo) over SOTA performance.