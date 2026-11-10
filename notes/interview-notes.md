# Interview notes

Talking points for this project. Every claim here points to something in the repository, so you can show it live.

## The 30-second version (say this out loud)

"Satellites photograph the same place again and again. I wanted a system that spots what changed between two
photos, like a forest being cleared, without needing thousands of hand-labelled examples. So I trained a neural
network on unlabelled photos to learn what imagery looks like, then compared how it describes the before and after
photos. My first test on photos it had never seen came out at chance level. I tracked it to one line of
preprocessing: I was stretching each photo's brightness separately, so a big change in one photo made the whole scene
look different. Sharing one stretch took the same trained network from 0.49 to 0.999. But when I added baselines I
found that an untrained network, and even a plain subtraction of the photos, do just as well on my synthetic data. So
I can't claim the pretraining helps yet, and the real test is real data."

If they ask for more, use the sections below.

## One-minute pitch

I built a pipeline that pretrains a small CNN on unlabelled Sentinel-2 imagery using contrastive learning (SimCLR-style),
then detects change by comparing before and after feature maps. The pipeline is complete: data ingestion, training,
tiled inference, thresholding, evaluation, a dashboard, tests and CI. When I measured it on held-out scenes it first
failed, I found and fixed the cause, and I then reported honestly that the fix does not show the pretraining helps.

## What I'd show

1. `python scripts/demo.py --synthetic`: the full pipeline in about a minute.
2. The project page: the before/after picture first. The "old scaling" image against the fixed one is the clearest
   part of the story.
3. `scripts/reproduce_results.py`: the held-out split, and every baseline computed on the same scenes.
4. `streamlit run app.py`: upload a pair and show the heatmap and mask.
5. CI: a green run, and `tests/test_pipeline.py` as the end-to-end test.

## Design decisions and why

- **Contrastive pretraining (NT-Xent) on single images.** Two augmented views of one patch are positives. This needs
  no change labels, which are expensive to produce. The trade-off is that the objective never sees before/after pairs.
- **Small CNN, not a pretrained ResNet or ViT.** It trains on a laptop CPU, and I can read the whole model in one
  file. A larger backbone would be the first thing to try if the small model proves to be the bottleneck.
- **Dense multiscale cosine distance, not pixel subtraction.** The hypothesis was that feature distance is less
  sensitive to small spectral changes. The held-out result does not support it on these scenes: a plain subtraction
  does as well.
- **Otsu thresholding.** It keeps inference label-free. Its weakness showed up in the results: for the network it picks
  a cut-off that over-marks changed pixels (precision 0.59), so its map is worse than its ranking.
- **Tiled inference with overlap averaging.** It bounds the memory of each forward pass. Whole scenes are still loaded
  into memory, which I documented after I found the original README claimed otherwise.
- **One shared brightness stretch for a before/after pair** (`normalize_pair`). Training uses a per-photo stretch on
  single images, so training and scoring differ. That mismatch did not hurt on synthetic data, but I have not tested
  training on shared-stretch pairs.

## The result, and how I'd explain it

On 4 held-out synthetic scenes (pretrained on 8 others):

| Method | AUROC | F1 |
| --- | ---: | ---: |
| Trained network, shared stretch (current) | 0.999 | 0.739 |
| Same trained network, old per-photo stretch | 0.491 | 0.181 |
| Same network, untrained (random weights) | 0.998 | 0.875 |
| Colour difference, shared stretch | 1.000 | 0.999 |
| Colour difference, per-photo stretch | 0.740 | 0.240 |
| Colour difference, raw values | 1.000 | 1.000 |

Things to say:

- The earlier figure of AUROC 0.98 was measured on the training scenes, so it was not evidence of generalisation. I
  withdrew it. The first thing to check in any ML result is what data it was measured on.
- The failure was a preprocessing bug, not a modelling problem. Rows 1 and 2 are the same trained weights, and only
  the scaling differs. Baselines are what made that visible: a plain colour difference showed the same collapse
  (1.000 raw, 0.740 stretched), which pointed at the scaling.
- I first guessed the cause from the code, then tested it, and one of my own tests was wrong: it was meant to use a
  shared stretch but did not, so it reproduced the per-photo numbers. I caught it because the two results were
  identical to the last digit. Worth telling: it shows checking results, not just producing them.
- **Pretraining is not shown to help.** The untrained network scores 0.998, with a better F1 than the trained one.
  Say it plainly, because self-supervised pretraining is the point of the project.
- **The synthetic task is too easy.** A plain subtraction gets a perfect score, so these scenes cannot separate the
  methods. Real change is not a flat colour patch on a smooth background.
- Next steps, in order: real labelled data, a pretraining sweep against the untrained network, a threshold calibrated
  on a validation region instead of Otsu, and training on shared-stretch pairs.

## Bugs I found and fixed

- **Per-photo brightness stretch.** Each date was scaled by its own percentiles, so a large change in one photo shifted
  the scale of the whole scene. Fixed with a shared stretch, with a regression test that checks unchanged pixels
  stay identical across the two dates.
- **Masks treated as imagery.** Ground-truth masks sat in the same folder tree as the images, so training could sample
  a 1-band mask as an image (crash), and pairing could use a mask as the "after" image. Reproduced, then fixed, with a
  test.
- **Cloud pixels as zeros in the composite.** Cloudy pixels were set to 0 before the median, so they pulled the
  composite toward dark values. Fixed by using NaN. A mutation test confirms the new test fails on the old code.
- **Dashboard map plotted wrong pixels.** Row and column indices were subsampled separately, which paired unrelated
  coordinates. Fixed by subsampling flat indices.
- **Demo skipped evaluation on real data.** The masks path did not match the fetch layout.
- **AUROC broke ties by pixel order.** Replaced with average ranks, and tested against the pairwise definition.
- **A diagram error on the project page.** Two nested elements both had the Mermaid class, so it read HTML as diagram
  text. I replaced the library with an inline SVG, so the page loads no scripts, and added tests that fail if one
  returns.

## Likely questions

**Did the pretraining help?** Not on this data, and I'd say so first. An untrained network does as well. I would test
it on harder, real data and sweep the pretraining budget before claiming anything.

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

**How do you know the code works?** Unit tests cover the I/O, metrics, tiling, scaling and composite logic. An
end-to-end test runs training, prediction and evaluation on tiny data. CI runs lint and tests on Python 3.10 and 3.12.
Results are reproducible with one command, and the demo's synthetic metrics matched to full precision before and after
my refactor, using a fixed seed.

**What would you do with another week?** Test on real labelled data first, since the synthetic scenes cannot tell the
methods apart. Then a pretraining sweep against the untrained network, a calibrated threshold, and the NDVI and
supervised baselines.

## Things I would not claim

- That pretraining helps. The untrained network does as well on this data.
- That the model detects real land-cover change. Nothing here measures that.
- That the synthetic numbers mean anything beyond a pipeline check.
- That tiled inference makes memory use independent of scene size. Scenes are loaded in full.
