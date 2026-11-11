# StreetSpatial V0: execution report

Date: 2026-11-11  
Status: 200-scene pilot completed; results are exploratory because 190 scenes
use detector-proposed boxes that have not yet received individual visual review.

## Executive summary

The end-to-end experiment ran on real Google Street View images with frozen
Qwen2.5-VL-7B-Instruct features. Token alignment and transformed labels passed
the automated checks. A scene-held-out linear probe recovers image-plane object
coordinates above the mean and random-feature baselines, especially at late
visual layers and LLM layer 14. The strongest original-view result is LLM 14:
coordinate R2 = 0.823 and MAE = 0.054 (three probe seeds, 128 PCA dimensions).

This establishes **linearly decodable spatial information**. It does not, by
itself, establish a dedicated or causal internal coordinate system. The direct
patch-position baseline has MAE about 0.007 and R2 about 0.995. It uses the
annotation box to select patches, as object feature pooling does; it is an
oracle geometry reference, not a model-only predictor. Patch location alone
nearly specifies the target. The current evidence supports the weaker claim
“decodable spatial information in Qwen representations”, not the stronger claim
“an independent internal spatial coordinate system”.

## Data and quality control

- Input: existing real Street View files from the 20,000-row
  `real20k_20261110/images.train.jsonl` manifest. Source images were not
  modified.
- Sampling: 200 distinct panorama scenes in the final curated set: Austin 66,
  Bogota 45, London 58, Lagos 31. All transformed views from one `scene_id`
  remain in one split.
- Annotation: 10 scenes were visually reviewed. The other 190 scenes use
  Grounding DINO proposals with conservative category names; uncertain
  unreviewed lamp-post proposals are labeled `Pole`.
- Dataset output: 1,398 valid views and 3,175 object-view records. The view
  manifest contains original, mirror, shift, and zoom variants; clipped or
  incomplete objects are skipped. Scene split is 120/40/40 (train/validation/
  test), with 1,905/629/641 object views in the main stage-wise analysis.
- Geometry checks: boxes, centers, mirror/shift relations, crop bounds, and
  scene grouping passed. Twelve unit tests pass.
- Model and preprocessing: Qwen2.5-VL-7B-Instruct and Grounding DINO tiny are
  pinned under `/host/mnt/nas/urban-visual-gene/streetspatial_v0/models/`.
  The extraction environment uses Transformers 4.56.2 and BF16 model forward
  passes; object caches are FP16.
- Token audit: every view has 154 image tokens; merger and LLM visual-token
  embeddings have maximum alignment error 0. All eight cached stages contain
  finite values and 1,398 records.

The complete audit is in [pilot200_audit.json](manifests/pilot200_audit.json).

## Analysis protocol

Stages are `encoder_patch`, `encoder_0`, `encoder_16`, `encoder_31`, `merger`,
`llm_0`, `llm_14`, and `llm_27`. Object features use masked mean pooling over
the verified visual-token boxes. For each stage, a standardized 128-component
PCA followed by ridge regression predicts normalized `(x, y)` centers. PCA and
probe fitting use original-view training scenes only; validation selects the
ridge penalty; the test set contains held-out scenes and all valid transforms.
Seeds 17, 42, and 73 resample training scenes. Tables report means over these
three probe seeds; standard deviations and scene-bootstrap confidence intervals
are stored in the CSV files where applicable. For example, LLM 14 on the 93
original-view test objects has seed-42 coordinate MAE 0.053 with scene-bootstrap
95% CI [0.046, 0.060] and R2 0.830 with CI [0.779, 0.857].

Results are in [metrics](metrics), with plots in [figures](figures).

## RQ1: coordinate decodability and geometric consistency

### Original-view coordinate probe

| Stage | Coordinate R2 | MAE |
|---|---:|---:|
| Visual encoder patch | -0.024 | 0.148 |
| Visual encoder 0 | 0.073 | 0.140 |
| Visual encoder 16 | 0.470 | 0.112 |
| Visual encoder 31 | 0.784 | 0.060 |
| Merger | 0.680 | 0.073 |
| LLM 0 | 0.689 | 0.072 |
| LLM 14 | **0.823** | **0.054** |
| LLM 27 | 0.689 | 0.078 |

The mean-coordinate baseline has MAE 0.146; random features have MAE about
0.149. Thus late encoder and LLM 14 features carry useful coordinate signal,
while patch and earliest-layer probes do not.

The direct patch-position baseline has MAE 0.0068 and R2 0.995 on the same
objects. Because its patch mask comes from the current annotation box, this is an
oracle geometry reference, not a deployable image-only baseline. It is the main
interpretation boundary: the probe shows recoverable location information, but
does not show a location code independent of the known object region.

### Equivariance and pair relations

Using the fixed original-view probe, LLM 14 has equivariance MAE 0.052 for
mirror, 0.037 for shift-left, 0.042 for shift-right, 0.043 for zoom-in, and
0.039 for zoom-out. Late encoder 31 is similar but weaker for mirror (0.053)
and shifts (0.034–0.064). For LLM 14, mirror-derived absolute-coordinate
predictions recover left/right order at 0.939 accuracy and above/below order at
0.833 accuracy; the corresponding relative-coordinate MAE is 0.065.

These transformation results are substantially better than early stages and
show geometric consistency of the decoded signal. They still do not distinguish
a learned coordinate representation from a robust readout of visual patch
position.

### Category transfer

Training on `Car` and `Tree` and testing on unseen `Pole` and `Traffic Sign`
shows partial transfer at LLM 14: test coordinate R2 averages 0.485 across
three seeds (MAE 0.070), versus 0.823 on the standard original-view test set.
The transfer test has only 24 held-out-category objects; early-stage R2 is
negative. Category-invariant decoding is therefore not established.

## RQ2: spatial and semantic organization

Spatial and semantic probe row spaces are partially overlapping, not orthogonal.
The subspace-overlap mean is 0.070 at LLM 14 (SD 0.042), 0.101 at encoder 31,
and 0.124 at the merger; early encoder patch overlap is 0.271. The decrease at
later stages suggests more structured separation, but subspace overlap is not a
test of statistical or causal independence.

At LLM 14, the spatial projection gives coordinate R2 0.817 and semantic
balanced accuracy 0.455. The semantic projection gives coordinate R2 0.234 and
semantic balanced accuracy 0.803. After removing the learned spatial probe row
space, coordinate R2 remains 0.661 while semantic balanced accuracy is 0.683.
The residual coordinate signal means the two-dimensional probe subspace is not
the whole spatial code; the result is compatible with entanglement and with a
distributed representation.

## RQ3: change across model stages

Coordinate information rises through the visual encoder (R2 -0.024 to 0.784),
drops at the merger (0.680), is retained at LLM 0 (0.689), peaks at LLM 14
(0.823), and drops again at LLM 27 (0.689). The same ordering appears in MAE
and in mirror/shift consistency. The merger therefore does not erase spatial
information, while the middle LLM stage is the strongest tested readout point.

Semantic and spatial organization also changes by stage: encoder 31 and LLM 14
have low overlap and high coordinate performance, whereas early encoder stages
have weak coordinate probes and larger overlap. These are task-space comparisons
within each stage; raw feature subspaces were not compared across dimensions.

## Sensitivity analysis: minimum original box side 20 px

Filtering to 129 scenes with at least two original objects whose shorter box side
is at least 20 pixels leaves 1,948 object views. Main trends remain: encoder 31
R2 = 0.766 (MAE 0.066), merger R2 = 0.625 (MAE 0.081), and LLM 14 R2 = 0.775
(MAE 0.060). The absolute values move with the object-size distribution, while
the late-stage advantage persists. Outputs are in [sensitivity](sensitivity).

## Answers to the research questions

1. **RQ1:** Yes, normalized object coordinates are linearly decodable from late
   Qwen stages and the decoded coordinates respond consistently to mirror,
   shift, and zoom transforms. The patch-position control prevents interpreting
   this as evidence for a dedicated coordinate system.
2. **RQ2:** Spatial and semantic information are partially separable but remain
   entangled. A spatial probe is sufficient for much of coordinate decoding, yet
   coordinate information remains after its row space is removed.
3. **RQ3:** Spatial readout strengthens from early visual layers to encoder 31,
   is partially compressed at the merger, peaks at LLM 14, and weakens at LLM
   27. This stage profile is stable under the large-object sensitivity check.

## Limitations and next actions

- 190 of 200 scenes still need individual box review. Until that is done,
  semantic/category-transfer and relation results are exploratory and may
  contain detector or category errors.
- The oracle patch-position baseline is extremely strong. A dedicated coordinate claim
  requires controls that remove or randomize absolute patch position while
  preserving object appearance, plus independent DINOv2/CLIP feature controls.
- The pilot does not justify causal patching. First audit all annotations,
  rerun the fixed analysis on the reviewed set, and add DINOv2/CLIP controls.
  If late-stage results replicate, the best candidates for optional causal
  validation are encoder 31 and LLM 14, with matched random-subspace and norm
  controls.
- The current 200-scene pilot is a feasibility study, not a final benchmark.
  Expanding to 500 scenes should wait until annotation review and the position
  control are complete.

## Reproduction

From `/root/code/urban-visual-gene/streetspatial_v0`:

```bash
bash scripts/run_v0.sh
```

The script runs geometry tests, view checks, resumable feature extraction,
cache audit, and the main analysis. The published metrics and figures are in
this repository. The original data, model snapshots, and feature caches remain
on the original host and are not part of this release; the script's absolute
paths must be adapted before reproducing on another machine. Existing source
images and user data were read-only inputs.
