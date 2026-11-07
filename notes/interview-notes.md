# Interview notes

Talking points for this project. Every claim here points to something in the repository, so you can show it live.

## The 30-second version (say this out loud)

"Satellites photograph the same place again and again. I wanted a system that spots what changed between two
photos, like a forest being cleared, without needing thousands of hand-labelled examples. So I trained a neural
network on unlabelled photos to learn what imagery looks like, then compared how it describes the before and after
photos. I tested it on photos it had never seen and compared it to simple baselines. It didn't beat them, and I
tracked down why: stretching each photo's brightness separately hides the very change I'm looking for. A plain
difference gets a perfect score without that stretch, and falls to 0.74 with it."

If they ask for more, use the sections below.

## One-minute pitch

I built a pipeline that pretrains a small CNN on unlabelled Sentinel-2 imagery using contrastive learning (SimCLR-style),
then detects change by comparing before and after feature maps. The pipeline is complete: data ingestion, training,
tiled inference, thresholding, evaluation, a dashboard, tests and CI. When I measured it on held-out scenes, it did not
beat a simple per-pixel spectral difference. I found that, reported it, and fixed the evaluation so the report is
honest.

## What I'd show

1. `python scripts/demo.py --synthetic`: the full pipeline in about a minute.
2. The project page: the before/after picture first, then the held-out table with the baselines.
3. `scripts/reproduce_results.py`: the held-out split, and the baselines computed on the same scenes.
4. `streamlit run app.py`: upload a pair and show the heatmap and mask.
5. CI: a green run, and `tests/test_pipeline.py` as the end-to-end test.

## Design decisions and why

- **Contrastive pretraining (NT-Xent) on single images.** Two augmented views of one patch are positives. This needs
  no change labels, which are expensive to produce. The trade-off is that the objective never sees before/after pairs.
- **Small CNN, not a pretrained ResNet or ViT.** It trains on a laptop CPU, and I can read the whole model in one
  file. A larger backbone would be the first thing to try if the small model proves to be the bottleneck.
- **Dense multiscale cosine distance, not pixel subtraction.** Feature distance is less sensitive to small spectral
  changes. This was the hypothesis, and the held-out result shows it does not yet hold on these scenes. See below.
- **Otsu thresholding.** It keeps inference label-free. Its weakness is that it picks a threshold per scene, so the
  threshold can vary a lot between scenes and seasons.
- **Tiled inference with overlap averaging.** It bounds the memory of each forward pass. The honest caveat is that
  whole scenes are still loaded into memory, which I documented after I found the original README claimed otherwise.
- **Per-image normalisation.** Simple and robust, but each date is scaled by its own percentiles. A change in scene
  composition can then shift both scales, which can cause false change. I kept it and documented it as a limitation.

## The result, and how I'd explain it

On 4 held-out synthetic scenes (pretrained on 8 others):

| Method | AUROC | F1 |
| --- | ---: | ---: |
| Trained encoder | 0.491 | 0.181 |
| Random-weight encoder | 0.702 | 0.529 |
| Colour difference, each photo stretched on its own | 0.740 | 0.240 |
| Colour difference, raw values (no stretch) | 1.000 | 1.000 |

Things to say:

- The earlier figure of AUROC 0.98 was measured on the training scenes, so it was not evidence of generalisation. I
  withdrew it. The first thing to check in any ML result is what data it was measured on.
- Adding baselines made the result legible. Without the random-encoder baseline I could not tell whether the learned
  features helped at all. The random encoder scoring above the trained one suggests the trained features are not
  adding value, and that the architecture's inductive bias is doing some of the work.
- The leading explanation now has evidence. Per-image normalisation stretches each photo by its own percentiles.
  A large change in one photo shifts its stretch, so the whole scene looks different. I tested this on the plain
  colour difference: skipping the stretch took AUROC from 0.74 to 1.00 on every held-out scene. The encoder is
  trained and run on stretched photos too, so it very likely has the same problem. I did **not** retrain it with a
  shared stretch, so say "very likely", not "proven".
- The other hypotheses, in the order I would test them after that: the pretraining budget is too small (8 scenes,
  30 epochs), and the contrastive objective learns features that are invariant to exactly the spectral differences
  I need.
- One caution: the raw difference scoring 1.000 shows the synthetic task is trivial, not that the method is good.
  Real change is not a flat colour patch on a smooth background.

## Bugs I found and fixed

- **Masks treated as imagery.** Ground-truth masks sat in the same folder tree as the images, so training could sample
  a 1-band mask as an image (crash), and pairing could use a mask as the "after" image. Reproduced, then fixed, with a
  test.
- **Cloud pixels as zeros in the composite.** Cloudy pixels were set to 0 before the median, so they pulled the
  composite toward dark values. Fixed by using NaN. A mutation test confirms the new test fails on the old code.
- **Dashboard map plotted wrong pixels.** Row and column indices were subsampled separately, which paired unrelated
  coordinates. Fixed by subsampling flat indices.
- **Demo skipped evaluation on real data.** The masks path did not match the fetch layout.
- **AUROC broke ties by pixel order.** Replaced with average ranks, and tested against the pairwise definition.

## Likely questions

**Why did the model fail?** I don't know yet, and I'd say so. I have three hypotheses above, and I know what to test
for each. The honest answer is more useful than a confident guess.

**Why not just use a pretrained model?** I'd try that next. It was out of scope for a CPU-only first version, and
a pretrained backbone would change the question from "does my pretraining help?" to "does transfer help?".

**How would you evaluate this for real?** Use public labelled change datasets, split by geography rather than by patch,
fit any threshold on a validation region, report precision-recall curves, and compare against spectral and index
baselines (NDVI, NDBI, NDWI), plus a supervised U-Net trained on the same scenes.

**What about seasonality and cloud cover?** Both are real confounds. Seasonal change in vegetation can look like land
change. Cloud masking is handled by the SCL band, but I have not evaluated its quality. Stratifying results by season
and cloud cover is on the list.

**Why CNN and not a transformer?** Data and compute. The dense feature stages are simple to read in a small CNN. A
transformer would be a reasonable next experiment at larger scale.

**How do you know the code works?** Unit tests cover the I/O, metrics, tiling and composite logic. An end-to-end test
runs training, prediction and evaluation on tiny data. CI runs lint and tests on Python 3.10 and 3.12. The synthetic
demo's synthetic metrics matched to full precision before and after my refactor, using a fixed seed.

**What would you do with another week?** First, retrain with one shared brightness stretch for both dates, since that
is the leading explanation for the failure. Then test on real labelled data, add the NDVI and supervised baselines,
and run a proper pretraining sweep over the number of scenes and epochs.

## Things I would not claim

- That the model detects real land-cover change. Nothing here measures that.
- That the synthetic numbers mean anything beyond a pipeline check.
- That tiled inference makes memory use independent of scene size. Scenes are loaded in full.
