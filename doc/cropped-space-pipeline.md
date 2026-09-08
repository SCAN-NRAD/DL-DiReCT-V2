# A single-grid (cropped-space) surface pipeline

Status: design + implementation, verified on sub-POBHC0002_TI900TR1700_run1
(crop 131x141x167) and sub-POBHC0001_TI900TR1700_run1. Numbers quoted below
are measured; where they are from one subject that is stated.

## 1. The problem, as measured

The FreeSurfer-free path (`dl+direct-nofs.sh`) juggles three frames:

| frame | grid | produced by | used by |
|---|---|---|---|
| **cropped** | e.g. 131x141x167, 1 mm, true (oblique) scanner affine | `crop.py` | `DeepSCAN` (`softmax_seg.nii.gz`, `seg_<Label>.nii.gz`), `DiReCT`/`direct_cuda` (velocity field, `T1w_thickmap.nii.gz`, `seg.nii.gz`), `extract_stats.py`, all of `field_pial_prototype.py` |
| **conformed** | 256^3 LIA, true affine recorded in `mri/conform_vox2ras.txt` | `preparedata.py` via `nibabel.processing.conform` | `dl_wm_surface_parallel_dev.py` (WM mask -> nighres -> `?h.white`), `surface_frames.volume_info_from_prep`, `compare_surfaces.py`, `extract_morphometrics.py` / `smooth_pial.py` (FreeSurfer path only) |
| **tkrRAS** | not a grid: a fixed relabelling of voxel indices, `T = [[-d,0,0,N d/2],[0,0,d,-N d/2],[0,-d,0,N d/2]]` built from `dim`/`pixdim` alone | every writer of a `.white`/`.pial` | every reader of a surface file (freeview included) |

Because tkrRAS is derived from the grid, **each grid has its own tkrRAS**.
The frame hazard is that two grids exist, not that tkrRAS exists.

### 1.1 What the conform actually does (sub-POBHC0002)

`conform()` uses `rescale_affine`, which maps input voxel `(S-1)//2` onto output
voxel `127`. The cropped->conformed voxel map measured from the two affines is

```
[[1 0 0 62.0003]
 [0 1 0 57.    ]
 [0 0 1 44.    ]]
```

an integer translation (the 3e-4 residual is the input's 0.99999475 mm x-zoom).
The stored `mri/aparc.atlas+aseg.nii.gz` equals `softmax_seg.nii.gz` shifted by
(62,57,44) on 99.884% of voxels; the 3576 mismatches are exactly the
corpus-callosum voxels preparedata relabels (3573 of them) plus 3 stragglers.
Nothing non-zero lies outside the shifted crop box. **There is no half-voxel
resampling in the conform itself**: it is a lossless pad + integer shift.

### 1.2 Where the half voxel really is

tkrRAS centres on `S/2`; conform centres on `(S-1)//2`. For an odd `S` these
differ by half a voxel. On this subject all three crop dims are odd, and

```
tkr_conformed(V) - tkr_cropped(v) = (+0.5, -0.5, +0.5) mm   for V = v + (62,57,44)
```

`field_pial_prototype.py` builds its `tovox` from the **cropped** `seg_*.nii.gz`
header and applies it to a white surface built in the **conformed** tkr frame.
Its own module docstring claims the two centrings "cancel exactly"; they do not
for odd dimensions. Measured: every white vertex is placed at
`(-0.5, -0.5, -0.5)` voxels from its true cropped-grid position (0.866 vox,
max deviation from that constant 2e-6), on both hemispheres. The consequences
are visible in the pipeline's own frame check:

| white vertices vs WM boundary | true frame | frame as used by field_pial |
|---|---|---|
| lh mean \|sd\| (seg==3) | 0.479 vox | 0.850 vox |
| rh mean \|sd\| (seg==3) | 0.461 vox | 0.838 vox |
| lh mean \|sd\| (surface WM mask) | 0.310 vox | 0.721 vox |

The pial propagation therefore samples the velocity field, the signed-distance
floor, the no-push masks and the sulcal sheet 0.87 voxels away from where the
white surface actually is. Whether that is "bad" for the pial is a separate
question (section 5 measures it); it is unambiguously not what the code says it
does.

### 1.3 The "two WM definitions" (contradicts the framing)

The surface is built from the aparc labels (`Left*`/`Right*` minus cerebellum
and hippocampus, subcortical structures filled); the pial floor is referenced
to `build_seg_maps`' `seg==3` from the WM logits. Near cortex these coincide:

| | count | signed distance to the *other* mask, mean / median | inside the other mask |
|---|---|---|---|
| cortex-adjacent boundary voxels of the surface WM | 147,912 | -0.930 / -1.000 | 96.4% |
| cortex-adjacent boundary voxels of `seg==3` | 147,523 | -0.996 / -1.000 | 99.2% |

(-1.0 is what a boundary voxel of a mask scores against its *own* mask.) Within
two voxels of cortex the symmetric difference is 5,937 + 1,240 voxels out of
~148k boundary voxels. The white vertices' **median** signed distance is +0.025
vox to `seg==3` and -0.040 vox to the surface mask: a 0.065-voxel disagreement,
not 0.7. The ~0.7-voxel figure matches the mean |sd| of the *misplaced* surface
(0.72-0.85 above), i.e. the frame error masquerading as a definition
difference. The **means** differ more (+0.20 vs -0.07) because the surface mask
includes the filled subcortical structures, over which the white surface sits
far outside `seg==3`; those vertices are already exempt from the tightened
floor via the no-push mask.

Which is "correct" for the floor: the mask the white surface was built from,
because the floor's purpose is "pial outside the white surface". In the
single-grid design that mask is available on the same grid with no resampling,
so the pial step could take it directly (the `--floor-mask` option, since removed
as measured useless -- section 5.15). It is an
option, not the default: it changes the propagation, and the measured
disagreement it corrects is 0.065 voxels.

## 2. Which conformed outputs are still load-bearing

With `mris_make_surfaces` gone, grep over the package for readers of `mri/`:

| file | readers outside `fast_surface_reconstruction.sh` | in nofs path? |
|---|---|---|
| `aparc.atlas+aseg.nii.gz` | `dl_wm_surface_parallel_dev.py`, `surface_frames.volume_info_from_prep`, `compare_surfaces.py`, `extract_morphometrics.py` | yes |
| `conform_vox2ras.txt` | `surface_frames.py`, `compare_surfaces.py` | yes |
| `norm.mgz` / `brain.mgz` | none in code; used for QC (freeview, WM/GM midpoint check) | QC only |
| `raw_brain.mgz` | `extract_morphometrics.py`, `smooth_pial.py` | not called by nofs |
| `aseg.presurf.mgz`, `filled.mgz`, `wm.seg.mgz`, `aparc.DKatlas+aseg.mgz` | FreeSurfer binaries only | no |

`extract_stats.py` reads `T1w_thickmap.nii.gz`, `seg.nii.gz`,
`softmax_seg.nii.gz` -- all cropped already. `extract_morphometrics.py` and
`smooth_pial.py` are **not invoked** by `dl+direct-nofs.sh` (they need
`?h.pial.raw`, `?h.pial-outer-smoothed`, `?h.curv.pial`, which only the
FreeSurfer path produces); both derive tkr from `mri/raw_brain.mgz`'s header
(`MGHHeader.get_vox2ras_tkr`, same formula as ours), so they keep working on a
cropped-grid `mri/` should anyone run them.

**nighres** (`topology_correction` + `levelset_to_mesh`) reads only `dims` and
`zooms` from the image it is given and returns voxel coordinates
(`use_resolutions=False`). It needs nothing about 256^3. The one thing that
could matter on a smaller grid is the object touching the volume border
(`propagation='background->object'`); measured, the WM masks sit >= 4 voxels
from every face of the crop (lh bbox [63,5,4]-[126,113,160], rh
[5,5,4]-[70,109,161] in 131x141x167), because the crop is the *brain mask's*
bounding box and WM is interior to it. `dl_wm_surface_parallel_dev.py` now
guards this anyway (pads by an integer margin and subtracts it).

Nothing in the FreeSurfer-free path is load-bearing for the 256^3 grid.

## 3. Design

**One grid: the cropped grid.** Surfaces are written in that grid's tkrRAS,
carrying `volume_info` that describes the cropped grid (dims, zooms, direction
cosines and `cras` from its true affine).

Why still tkrRAS and not scanner RAS: the FreeSurfer surface format has no
way to say "these coordinates are scanner RAS". Every reader (freeview,
`mris_*`, nibabel users following FreeSurfer's convention) interprets the
coordinates as tkrRAS of the geometry in `volume_info`, or of whatever volume
is loaded when `volume_info` is absent, and maps them to scanner RAS as
`vox2ras_scanner @ inv(vox2ras_tkr)`. For an oblique acquisition such as this
subject (8.5 degrees), no `volume_info` can make tkr equal scanner RAS
(tkr's direction cosines are fixed). Measured: writing scanner RAS would put
the vertices a mean 7.4 mm (lh) / 7.7 mm (rh) from where freeview would draw
them, plus the rotation. Writing tkrRAS-of-the-cropped-grid with truthful
`volume_info` is what makes freeview place them on `T1w_norm.nii.gz`, the
cropped T1, or any co-registered volume. tkrRAS of a *single* grid involves no
resampling and no second grid, which is what the hazard was.

Consumers that need scanner RAS keep using `surface_frames.our_surface_to_world`
(`conform_vox2ras.txt @ inv(tkr(dims, zooms))`), which is correct for both the
old and the new outputs because each reads its own `mri/`.

### Changes (all additive; the conformed path is byte-identical)

* `preparedata.py --space cropped`: no conform. Writes
  `mri/aparc.atlas+aseg.{nii.gz,mgz}`, `mri/aparc.DKatlas+aseg.mgz`,
  `mri/conform_vox2ras.txt` (now the cropped affine; name kept because its
  readers define it as "vox2ras of the grid `aparc.atlas+aseg` is on"),
  `mri/norm.mgz`, `mri/brain.mgz`, `mri/raw_brain.mgz` -- all on the cropped
  grid with the true affine. **Dropped**: `aseg.presurf.mgz`, `filled.mgz`,
  `wm.seg.mgz` (FreeSurfer-binary inputs only). The midline split's
  hard-coded `255` upper bound becomes `shape[0]` (output-identical on 256^3).
* `dl_wm_surface_parallel_dev.py`: **no frame change** -- it derives tkr from
  `mri/aparc.atlas+aseg.nii.gz`, so it follows whatever grid preparedata
  wrote. Added: an integer border-pad guard around nighres.
* `field_pial_prototype.py`: **no frame change** -- `make_transforms(ref_img)`
  was always the cropped grid's tkr; the surfaces now really are in it.
  Added: (a) a hard check in `main()` that the white surface's `volume_info`
  dims match the reference grid, which catches the exact mismatch of 1.2
  (surfaces without `volume_info` get a warning, not a failure, since every
  pre-existing output lacks it); (b) the module
  docstring's "Coordinate frames" section rewritten to state the measured
  offset rather than the cancellation claim. Verified: the legacy combination
  (old `mri/` + old white) still reproduces the stored batch pial to 0.00 mm.
* `dl+direct-nofs.sh --cropped-space`: runs `preparedata.py --space cropped`.
  Default unchanged (conformed) so existing invocations are untouched; the
  recommendation is to switch the default once the parent is satisfied with
  section 5.
* `surface_frames.py`: docstrings updated; code unchanged.

## 4. Migration / compatibility

* Old outputs: surfaces in the 256^3 tkr frame, `mri/` conformed. New
  outputs: surfaces in the cropped tkr frame, `mri/` cropped. The two tkr
  frames differ by a constant per-axis offset of 0 or +/-0.5 mm (parity of
  each crop dim) plus the (62,57,44)-type integer shift, so **never compare
  old and new surfaces by tkr coordinates**; go through scanner RAS with
  `our_surface_to_world`, which handles both (verified: identical midpoint
  numbers from either grid, section 5).
* A run directory mixes frames if `preparedata` is re-run in a different
  `--space` without re-running the surface step. The `volume_info` check in
  `field_pial_prototype.main` refuses that combination.
* `extract_morphometrics.py` / `smooth_pial.py`: unchanged and consistent with
  a cropped `mri/` (both take tkr from `raw_brain.mgz`); they still need
  FreeSurfer-path inputs that nofs does not produce.
* Old field_pial outputs cannot be reproduced bit-for-bit by the new path:
  they were propagated from a white surface displaced 0.87 vox (section 1.2).
  The old *white* surfaces are reproduced (section 5).
* freeview: the memory note "Did not find any volume info ... expected" is
  obsolete for new outputs; they carry volume geometry.

## 5. Verification

### 5.1 Setup

Two completed batch runs, read-only: **S1** = sub-POBHC0002_TI900TR1700_run1
(crop 131x141x167, all dims odd) and **S2** = sub-POBHC0001_TI900TR1700_run1
(129x142x172, parity [1,0,0]). New outputs were written to a scratch copy that
symlinks the inputs. White surfaces: `preparedata.py --space cropped` then the
unchanged `dl_wm_surface_parallel_dev.py` (nighres on the cropped grid: 17 s
wall for both hemispheres). Pials: the saved `field_pial/best_Velocity.nii.gz`
-- a cropped-grid quantity that does not depend on the surface frame -- fed to
`propagate()` with `build_no_push_mask`/`build_pin_mask` exactly as
`run_pipeline` calls them (no GPU solve). Two checks that the harness is
faithful:

* re-propagating the batch's own white, in the batch's (misplaced) frame,
  reproduces the stored `?h.pial.field_best` to **0.00000 mm** max, both
  subjects, both hemispheres;
* the real CLI (`field_pial_prototype.main()` with only
  `solve_velocity_field` replaced by the saved field) reproduces the driver to
  0.00e+00 on the new outputs, and reproduces the stored batch pial to
  0.00e+00 on the old layout (old `mri/` + old white, "legacy" combination) --
  i.e. the default path is bit-identical after these changes.

Three positions are compared, all mapped to scanner RAS via
`our_surface_to_world` of the respective run:

* **A** old white, propagated in the frame field_pial used (the batch result);
* **B** old white, placed correctly in the cropped frame, then propagated;
* **C** new white (built on the cropped grid), propagated.

### 5.2 White surface: old vs new

| | S1 lh | S1 rh | S2 lh | S2 rh |
|---|---|---|---|---|
| vertices old / new | 144536 / 144536 | 144786 / 144848 | 135550 / 135550 | 134650 / 134656 |
| naive tkr difference new-old (mm) | (-0.50, +0.50, -0.50) | -- | (-0.50, 0.00, 0.00) | -- |
| point-to-surface old->new, mean / median / 95% (mm) | 0.0012 / 0.00008 / 0.0048 | 0.0015 / 0.00007 / 0.0046 | 0.0010 / 0.00006 / 0.0044 | 0.0012 / 0.00006 / 0.0044 |
| vertices >0.1 mm / >0.5 mm from the other surface | 97 / 10 | 258 / 35 | 30 / 5 | 122 / 12 |
| max (mm) | 0.78 | 1.13 | 0.90 | 0.71 |
| `check_frame_alignment` mean \|sd\|, new white in cropped frame (vox) | 0.479 | 0.461 | 0.469 | 0.474 |
| same statistic, old white as field_pial read it | 0.850 | 0.838 | -- | -- |
| `check_surface_frame` (tissue %, bg %) | 96.2, 1.0 | 96.4, 0.9 | 96.4, 0.9 | 96.3, 0.9 |

The naive tkr difference is exactly the parity prediction (section 1.2). Index-
wise comparison is meaningless once a vertex count differs (an inserted vertex
shifts every later index), hence point-to-surface.

### 5.3 The residual is nighres, not the frame

`topology_correction` is exactly deterministic (two runs on the same array:
identical points and faces; forcing the 0.99999475 x-zoom to 1.0: identical),
and re-running it on the 256^3 array reproduces the stored old rh.white
vertex/face counts exactly. But it is **not invariant to the zero padding
around the object**. S1 rh, same mask, corrected-object voxels that differ from
the unpadded (131x141x167) run:

| padding | shape | verts | object voxels | differing voxels |
|---|---|---|---|---|
| 0 | 131x141x167 | 144848 | 264104 | 0 |
| 1 each side | 133x143x169 | 144788 | 264081 | 41 |
| 2 | 135x145x171 | 144794 | 264079 | 35 |
| 8 | 147x157x183 | 144784 | 264080 | 40 |
| (62,57,44) = the conform | 256^3 | 144786 | 264079 | 35 |
| 1 voxel on z only | 131x141x169 | 144786 | 264079 | 39 |

35-41 of 264k object voxels (0.015%), all boundary voxels along the medial
side of the hemisphere, flip in or out depending on the array around them.
That is the entire source of the >0.1 mm tails in 5.2. The mechanism inside
nighres was not investigated; the numbers say it is tie-breaking-scale noise
of the correction, present between ANY two paddings, not something the
cropped grid introduces. It is why `dl_wm_surface_parallel_dev.py` pads only
when the object touches a face (never, for a brain-mask crop).

### 5.4 Pial surface

| (mm) | S1 lh | S1 rh | S2 lh | S2 rh |
|---|---|---|---|---|
| **A vs B** (same white, frame fixed), vertex-wise mean / median / 95% | 0.50 / 0.48 / 0.95 | 0.50 / 0.49 / 0.92 | 0.29 / 0.29 / 0.54 | 0.29 / 0.28 / 0.54 |
| A vs B, vertices moved >0.1 mm / >0.5 mm | 94.3% / 47.3% | 94.5% / 48.3% | 91.4% / 7.4% | 90.6% / 7.4% |
| **B -> C** (old vs new white, both correct), point-to-surface mean / median / 95% | 0.0006 / 0.00004 / 0.0017 | 0.0008 / 0.00003 / 0.0018 | 0.0005 / 0.00003 / 0.0017 | 0.0006 / 0.00003 / 0.0017 |
| stored old pial -> new pial C, point-to-surface mean / median | 0.27 / 0.25 | 0.28 / 0.26 | 0.17 / 0.16 | 0.17 / 0.15 |

So: the new path's pial is where the old path's pial would have been had the
white been read in the right frame (B vs C ~0.0006 mm), and it is 0.5 mm (S1,
three odd dims) / 0.3 mm (S2, one odd dim) away from where the old path
actually put it. Self-consistency metrics, S1 (S2 in the same direction):

| | A lh | B lh | C lh | A rh | B rh | C rh |
|---|---|---|---|---|---|---|
| transit_pct | 2.54 | 2.99 | 2.99 | 2.33 | 2.84 | 2.83 |
| end_in_wm_count | 47 | 71 | 66 | 66 | 86 | 85 |
| flipped_face_pct | 0.071 | 0.035 | 0.035 | 0.097 | 0.052 | 0.051 |
| self_intersections | 2889 | 3029 | 3017 | 2682 | 2533 | 2527 |
| mean_displacement_mm | 3.262 | 3.255 | 3.255 | 3.235 | 3.234 | 3.233 |

A->B is a controlled change of one variable (the half-voxel placement):
transit and end-in-WM go up, flipped faces go down, self-intersections mixed.
Why the metrics move the way they do is not established here. Note the
metrics' own reference (`seg`, the no-push masks) was also displaced in A, so
A's numbers are not measurements of the same thing as B's.

### 5.5 White surface at the WM/GM intensity midpoint

Sampling each run's own `norm.mgz` along the vertex normal (0.05 mm steps,
midpoint = mean of the WM and cortex median intensities, nearest crossing to
the vertex):

| signed offset to the midpoint crossing (mm), mean / median | old (256^3 norm) | new (cropped norm) |
|---|---|---|
| S1 lh | +0.248 / +0.175 | +0.248 / +0.175 |
| S1 rh | +0.215 / +0.175 | +0.215 / +0.175 |
| S2 lh | +0.273 / +0.225 | +0.273 / +0.225 |
| S2 rh | +0.248 / +0.225 | +0.248 / +0.225 |

Identical to three decimals between grids (the two `norm.mgz` are the same
data on different grids, sampled through each surface's own frame). Sampling
the cropped norm through scanner RAS with the OLD surface gives the same
numbers again (S1 lh +0.249 / +0.150 at 0.1 mm steps), which is the exactness
of the world-coordinate route. These values are larger than the +0.12/+0.03 mm
quoted earlier in this project; that was a different estimator, and the claim
here is only old == new.

### 5.6 No half-voxel resampling remains

* The 256^3 conform is not run. (Where it is, in the default path, it is an
  integer shift -- section 1.1 -- so it never was a resampling error; the error
  was reading one grid's tkr coordinates in the other's frame.)
* nighres returns voxel coordinates on the cropped grid; the tkr affine is an
  exact linear map of those. Taubin smoothing acts on the mesh.
* `propagate()` samples the velocity field, `seg`, the signed distance and the
  no-push masks on the grid the white surface's coordinates refer to, which the
  `volume_info` check in `main()` now enforces.
* `volume_info` written into the surfaces reconstructs the cropped affine to
  6e-9 and gives a FreeSurfer-convention reader the same surface->scanner map
  as `our_surface_to_world` to 4e-10.
* The thickness map, `seg.nii.gz` and `extract_stats.py` were on the cropped
  grid already and are untouched.

### 5.7 freeview

Headless renders (coronal, centre slice, zoom 1.6) of the new surfaces on
three different grids: the uncropped oblique `T1w_norm.nii.gz` (208x256x240,
with skull), the cropped `norm.mgz`, and a 256^3 conformed `norm.mgz`. All
three place the surfaces on the anatomy, and freeview no longer prints "Did
not find any volume info" for them. Pixel comparison on the 256^3 volume
(same slice, same background):

| render pair | exact overlap | within 1 px | best shift |
|---|---|---|---|
| old white vs new white, 256^3 norm with a truthful header | 98.9% | 100.0% | (0, 0) |
| old pial vs new pial, same volume | 39.8% | 70.5% | (-2, -2) px |
| old pial vs new pial, the batch's own `mri/norm.mgz` | 13.9% | 31.1% | (-3, +2) px |

The white is placed identically across the two frames. The pial differs by the
0.5 mm of 5.4 (about 2 px at this zoom). The last row is a caveat for old
outputs: the batch's `mri/*.mgz` were written by commit 195161a with the tkr
affine as their vox2ras (cras = 0), which HEAD's 8ce587e already fixed; a
surface that truthfully states its position cannot be displayed correctly on a
volume that misstates its own. Re-running `preparedata.py` (either space) on
an old run fixes the volumes.

### 5.8 `--floor-mask surface` (option since removed)

S1, saved field, `main()`: surface-mask floor vs `seg==3` floor, vertex-wise
0.013 mm mean (median 0.002, 95% 0.06, max 1.8), 2.7% (lh) / 2.8% (rh) of
vertices moved >0.1 mm. Metrics within noise of each other (lh transit 2.976
vs 2.986, end_in_wm 75 vs 66, flipped 0.039 vs 0.035; rh 2.821 vs 2.828, 88 vs
85, 0.053 vs 0.051). Consistent with section 1.3: the two WM definitions are
0.07 voxels apart in cortex, and the option is a small, optional correction.

### 5.9 End-to-end runs with a fresh GPU solve

Added after incorporation (commits 1e9297e, 227e7a6), closing the first item
of what 5.1 could not do. `dl+direct-nofs.sh --bet --cuda-direct --skip-naive`
run from the T1 on both subjects, so bet, the segmentation, the CUDA DiReCT
solve and all four steps ran again; 118 s and 114 s.

Frame check on the fresh white surfaces (mean |distance to the WM boundary|,
voxels) -- reproducing 5.2 exactly, against 0.850 / 0.838 for the old path:

| | lh | rh |
|---|---|---|
| sub-POBHC0002 (131x141x167) | 0.479 | 0.461 |
| sub-POBHC0001 (129x142x172) | 0.469 | 0.474 |

Stored conformed batch -> fresh cropped run, true point-to-surface (mm),
compared with 5.4's saved-field row in brackets:

| | lh white | lh pial | rh white | rh pial |
|---|---|---|---|---|
| sub-POBHC0002 | 0.0012 | **0.275** [0.27] | 0.0015 | **0.279** [0.28] |
| sub-POBHC0001 | 0.0010 | **0.173** [0.17] | 0.0012 | **0.168** [0.17] |

So a full fresh solve reproduces the saved-field result to two decimals.
Confirmed directly: the freshly solved `best_Velocity.nii.gz` is bit-identical
to the batch's (max |dv| 0 over the whole 131x141x167x3 field), the fresh
conformed white is bit-identical to the batch's, and `conform_vox2ras.txt`
matches to 0.0 -- nothing upstream of the frame changed between 195161a and
HEAD.

**Reproducibility.** The whole pipeline was run twice from the T1 on both
subjects: white and pial surfaces are bit-identical (0.0000 mm, all four
hemispheres), CUDA solve included. Run-to-run variance is zero, so none of the
differences above is noise.

Two measurement notes, both of which cost a wrong number first:

* Nearest-*vertex* distance is not point-to-surface. A KD-tree over the target's
  vertices gives 0.42 / 0.28 mm where point-to-triangle gives 0.275 / 0.173, because
  the nearest point usually lies inside a face. Use `trimesh.proximity.closest_point`.
* Vertex-wise comparison is invalid across a nighres re-run even when the vertex
  counts match: the median white difference is 0.0015 mm but the mean is 5.98 mm
  and the max 150 mm, i.e. a subset of vertices is *reordered*. Index correspondence
  survives propagation (same white, same field) but not topology correction.

### 5.12 WM taken from the white surface (now the default)

The velocity field was solved on `seg == 3` while the propagation started from
the white surface, which is a topology-corrected level set of a *different*
mask. Pointing only the floor at the surface (the removed `--floor-mask mesh`) changes
nothing, because the field it constrains still has its inner boundary
elsewhere. `--wm-from-surface` (reconcile_seg_with_surface) instead makes
`seg == 3` *be* the rasterized interior of the white surfaces, so the solve,
the floor and the escape rule share one definition.

sub-POBHC0002_TI900TR1700_run1, 6430 voxels demoted to cortex and 78884
promoted (ventricles, subcortical grey, CC fill):

| lh / rh | default | --wm-from-surface |
|---|---|---|
| self-intersections | 3017 / 2527 | **2324 / 2074** (-23% / -18%) |
| flipped faces % | 0.0346 / 0.0511 | **0.0266 / 0.0349** |
| mean displacement mm | 3.255 / 3.233 | 3.260 / 3.241 |
| pial verts inside the white surface | 600 / 641 | 646 / 675 |

The gain is not bought with travel. `end_in_wm` and `transit_pct` are NOT
comparable across this option: both are defined against `seg == 3`, which the
option redefines. The penetration it was aimed at is not fixed.

Confirmed visually by the user on the surfaces at RAS [11.8, -64.3, -10.1] and
made the default in `dl+direct-nofs.sh` (`--seg-wm` opts out).

**Outstanding.** One subject. This project's standard for a configuration
change is sign consistency across 36 hemispheres, which has not been run. Nor
has the effect on the reported DiReCT thickness been measured, and promoting
78884 ventricle/subcortical voxels to WM changes the thickness map -- the check
that rejected sigma=0.35, which won every surface metric while thinning cortex
by 0.34 mm.

### 5.13 36-hemisphere paired tests of the two changed defaults

Phantom of Bern, 18 scans / 36 hemispheres, paired within subject. Arms:
A = current default (--wm-from-surface, escape off); B = escape re-enabled,
reusing A's field via --velocity-from; C = WM from seg==3, own solve. Judged by
sign consistency, this project's standard.

**Escape (A vs B).** Confirmed inert and now removed from the default.

| metric | OFF (default) | ON | OFF better |
|---|---|---|---|
| end_in_wm | 373.0 | 373.5 | 14/36 (14 ties) |
| pial inside white | 656.0 | 656.5 | 8/36 (18 ties) |
| self-intersections | 1869 | 1864 | 12/36 (8 ties) |
| flipped faces % | 0.0319 | 0.0302 | 2/36 (5 ties) |
| vertex-travel thickness | 3.2416 | 3.2415 | +0.0001 mm |

Only flipped faces move (29/36 favour ON, ~5% relative), and that sign is built
from many tiny wins plus one 19% LOSS on sub-POBHC0001_TI800TR1700_run1 lh --
the largest single effect in either direction. Dropped.

**WM from surface (A vs C).** A genuine trade, not a free win.

| metric | from surface | from seg==3 | surface better |
|---|---|---|---|
| self-intersections | 1869 | 2287 | **36/36** (-18%) |
| flipped faces % | 0.0319 | 0.0373 | 29/36 |
| crossed CSF sheet | 2613.5 | 2573 | 24/36 |
| pial inside white | 656 | 555.5 | **3/36** (+18%) |
| thickness, vertex travel | 3.2416 | 3.2333 | +0.0084 mm, 36/36 thicker |
| thickness, volumetric DiReCT | 2.7108 | 2.7566 | **-0.0458 mm, 18/18 thinner** |

18% fewer tangles for 18% more pial-inside-white, plus a systematic 0.046 mm
thinning of reported DiReCT thickness -- comparable to this dataset's own
scan-rescan repeat difference (0.0506 mm), though 7x smaller than the 0.34 mm
that disqualified sigma=0.35. The two thickness definitions move in OPPOSITE
directions here; no evidenced explanation.

### 5.14 The two thickness definitions are not interchangeable per vertex

Same surfaces, 36 hemispheres, 5,040,290 vertices:

| definition | mean | sd across hemispheres |
|---|---|---|
| vertex travel, \|P_i - W_i\| | 3.2416 mm | 0.0448 |
| symmetric nearest neighbour | 2.5998 mm | 0.0325 |
| symmetric nearest, exact point-to-surface | ~2.574 mm | -- |

Nearest is smaller in 36/36 (it must be: the corresponding vertex is always a
candidate), by ~20%. Nearest-vertex overstates exact point-to-surface by only
1.00% at this separation. Vertex-wise Pearson r = 0.881 over all vertices,
**0.833 excluding pinned** (so ~31% of the variance in one is unexplained by the
other); per-vertex travel-minus-nearest has median 0.493 mm and 95th percentile
1.796 mm. But across the 36 hemisphere MEANS r = 0.967.

Regression slope is 0.614 with a ~0.61 mm intercept, not 0.8 through the origin,
so travel overstates more in thick regions: converting between the definitions
by one factor would distort regional patterns. Consequence: global and
hemisphere-level comparisons are insensitive to the choice (every ablation above
would reach the same verdict either way), vertex- and parcel-level work is not,
and the nearest-neighbour definition is the one comparable with FreeSurfer.

### 5.15 `--floor-mask` removed

Dropped entirely, together with `propagate(wm_mask=)` and
`run_pipeline(floor_wm_mask=)`. It let the pial's floor and escape rule use a
WM mask other than `seg == 3` -- the aparc labels the white surface was built
from ('surface'), or a rasterization of the white surface itself ('mesh').

It was the direct fix for the pial visibly cutting into the white surface, and
it did not work: penetration 600 -> 597 (lh) and 641 -> 681 (rh). Two further
reasons to remove rather than keep it: `--wm-from-surface` makes `seg == 3`
*be* the surface interior, so the floor already references the surface and the
option is near-redundant; and only ONE of the four `wmb = (seg == 3)` sites
honoured the override, so the sulcal sheet, the constrained-start relaxation
and `enforce_floor` silently disagreed with the propagation's floor whenever it
was used.

`rasterize_mesh`, `rasterize_mesh_pv` and `reconcile_seg_with_surface` are
kept -- they serve `--wm-from-surface` and the pial-inside-white metric.
Verified: with the option gone the default path reproduces the stored pial to
0.0000000000 mm on both hemispheres.

### 5.16 Defaults: WM from the surface, at partial volume

`--wm-from-surface` and `--wm-pv 3` are now the defaults; `--seg-wm` and
`--wm-pv 0` opt out. `--wm-from-surface` is accepted and ignored so existing
invocations keep working. Verified: with no flags the pial reproduces the
explicit `--wm-from-surface --wm-pv 3` run to 0.0000000000 mm on both
hemispheres.

Set on the user's decision. What is on the record for each, and what is not:

* `--wm-from-surface`: 36/36 fewer self-intersections (-18%) and 29/36 fewer
  flipped faces, against 33/36 MORE pial-inside-white (+18%) and a systematic
  -0.0458 mm shift in volumetric DiReCT thickness (18/18), comparable to this
  dataset's own scan-rescan repeat difference of 0.0506 mm (section 5.13).
* `--wm-pv 3`: one subject only. -16%/-15% self-intersections with labels held
  fixed, but with ~10% fewer sulcal-CSF crossings on a provably unchanged sheet
  and 0.053 mm thinner on the nearest-neighbour definition -- the same
  signature that, for the normal gate, turned out to be the pial not
  descending into sulci rather than a better surface. Not resolved either way;
  no 36-hemisphere paired test.

### 5.17 Dead code removed

`feathered_floor` (measured worse, never enabled) and `enforce_floor` (written,
measured, never wired into any code path) deleted: 93 lines, module 2285 ->
2192. Consequence worth recording: `enforce_floor` was the last untested lead
on pial-inside-white, so that defect is now parked rather than solved. Every
candidate has failed -- the floor mask was not the cause (5.15),
`--wm-from-surface` makes it worse (5.13), and floor strength does nothing for
it in rh (the floor sweep is flat from 0.0 to 1.5). `rasterize_mesh` survives,
so the defect can be re-quantified whenever it matters.

Still dead but not deleted: `untangle_by_separation`, `smooth_retraction`,
`taubin_guarded`, `mesh_roughness`, and the CLI-unreachable `repair_from_round`,
`retract_from_round`, `escape_on_magnitude`, `edge_floor`, `face_floor`,
`return_trajectory`.

### 5.10 Still not done

* Two subjects, same scanner session series. The parity rule is exact, so
  other crops differ only in which axes carry the 0.5 mm.
* No paired 36-hemisphere comparison of the surface-quality metrics between the
  two paths; 5.4's self-consistency table is two subjects.

### 5.11 The default was switched

`dl+direct-nofs.sh` now defaults to `--space cropped` (227e7a6). This was not
optional: `main()`'s `volume_info` guard refuses a 256^3 white surface against a
cropped reference grid, so a *fresh* conformed run dies at step 4 with

    lh.white was built on a [256, 256, 256] grid but the reference
    (seg_<Label>.nii.gz) grid is [131, 141, 167] ...

5.1's bit-identity check did not catch this because it re-ran the CLI over the
*stored* batch outputs, whose surfaces predate `volume_info` and so take the
guard's warning branch instead. `--conformed-space` restores the old grid for
emitting a 256^3 white surface for another tool, and is only usable with
`--skip-pial`; `--cropped-space` is accepted and ignored.
