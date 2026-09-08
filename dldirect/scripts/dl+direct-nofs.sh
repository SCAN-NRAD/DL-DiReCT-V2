#!/bin/bash
#
# dl+direct-nofs -- surface reconstruction with no FreeSurfer calls at all.
#
# fast_surface_reconstruction.sh runs five FreeSurfer binaries, but only to
# build FreeSurfer's *own* pial surface via mris_make_surfaces. The white
# surface it produces depends on none of them: preparedata.py and
# dl_wm_surface_parallel_dev.py together were verified to emit byte-identical
# lh.white and rh.white with FREESURFER_HOME unset and no FS binaries on
# PATH. This script keeps exactly those two steps and then propagates the
# white surface along the DiReCT velocity field to obtain the pial, which is
# what field_pial_prototype does.
#
# Measured: ~3.5 min/scan here vs ~11.6 min/scan for the FreeSurfer path.
#
# Remaining external dependency is nighres (topology correction), not
# FreeSurfer. See doc: dldirect/field_pial_prototype.py module docstring.
#
# Usage: dl+direct-nofs.sh [dl+direct options] T1_FILE OUTPUT_DIR

usage() {
cat << EOF
Usage: dl+direct-nofs [dl+direct options] T1_FILE OUTPUT_DIR

Runs the FreeSurfer-free surface pipeline:
  1. dl+direct.sh --no-fsr --keep   (conform, bet, segment, crop, DiReCT)
  2. preparedata.py                 (conform to LIA 256^3, CC->WM, filled/wm.seg)
  3. dl_wm_surface_parallel_dev.py  (nighres topology correction -> {lh,rh}.white)
  4. field_pial_prototype           (velocity-field propagation -> pial)

Options are passed through to dl+direct.sh; --no-fsr, --keep and --no-cth are
always added. --no-cth is what stops DL+DiReCT from being solved twice: the
pial step runs its own kelly_kapowski solve, and every solve already returns a
thickness map, so the separate DiReCT.py run is redundant. See --skip-pial
below for the one case where it is not. Typical invocation:

  dl+direct-nofs.sh --bet --cuda-direct -s SUBJ input_T1w.nii.gz out_dir

Extra options handled here:
  --sigma VALUE           smoothing_sigma for the best run (default 1.0, matching the
                          shipped solver; 0.35 was the surface-tested best)
  --dip-threshold VALUE   sulcal CSF sheet dip threshold (default 0.95)
  --skip-pial             stop after the white surface (steps 1-3 only). Since the
                          thickness map then has no solve to come from, DiReCT is
                          left enabled in this mode; pass --no-cth as well to drop
                          the thickness map too.
  --skip-naive            skip the naive baseline pial (halves the pial step: the
                          naive and best configurations differ in the velocity solve
                          itself, so the baseline is a second full GPU solve)
  --conformed-space       build the surfaces on a 256^3 LIA conform instead of the
                          cropped segmentation grid (the default). The conform exists
                          only for mris_make_surfaces, which this script does not run,
                          and it puts the white surface in a tkrRAS frame that differs
                          from the pial step's by half a voxel per odd crop dimension --
                          so the pial step REFUSES the resulting surfaces. Use this only
                          with --skip-pial, to emit a 256^3 white surface for another
                          tool. See doc/cropped-space-pipeline.md.
  --cropped-space         accepted and ignored; this is now the default.
  --seg-wm                take the pial step's white matter from the segmentation
                          (seg==3) instead of from the white surface's own interior,
                          which is the default. The default resolves that boundary at
                          partial volume (--wm-pv 3 in the pial step), so the solve's
                          priors carry sub-voxel occupancy rather than a 0/1 staircase.
                          See doc/cropped-space-pipeline.md.
EOF
	exit 0
}

die() {
	RET=$?
	echo "ERROR (${RET}): $1"
	exit 1
}

SCRIPT_DIR=`dirname $0`/..
PASSTHROUGH=()
SIGMA=1.0
DIP_THRESHOLD=0.95
SKIP_PIAL=0
SKIP_NAIVE=0
NO_CTH=0
SUBJECT_ID="subj_id"
PREP_OPTS=(--space cropped)
WM_FROM_SURFACE=1

while [[ $# -gt 2 ]] ; do
	case "$1" in
		-h|--help)		usage ;;
		--sigma)		SIGMA=$2 ; shift ;;
		--dip-threshold)	DIP_THRESHOLD=$2 ; shift ;;
		--skip-pial)		SKIP_PIAL=1 ;;
		--skip-naive)		SKIP_NAIVE=1 ;;
		--cropped-space)	;;  # the default; kept so old invocations still work
		--conformed-space)	PREP_OPTS=() ;;
		--seg-wm)		WM_FROM_SURFACE=0 ;;
		-f|--no-fsr)		;;  # implied
		-k|--keep)		;;  # implied
		-n|--no-cth)		NO_CTH=1 ;;  # implied unless --skip-pial
		# needed here for extract_stats.py below, and still passed through
		-s|--subject)		SUBJECT_ID=$2 ; PASSTHROUGH+=("$1" "$2") ; shift ;;
		*)			PASSTHROUGH+=("$1") ;;
	esac
	shift
done

[[ $# -eq 2 ]] || usage

T1=$1
DST=$2

# The intermediate seg_<Label>.nii.gz logits are the propagation's input, so
# --keep is not optional here. --no-fsr skips fast_surface_reconstruction.sh
# entirely, which is the whole point. --no-cth skips DiReCT.py: nothing between
# here and the pial step reads its output (preparedata.py needs softmax_seg and
# the cropped T1, dl_wm_surface_parallel_dev.py needs mri/aparc.atlas+aseg, and
# field_pial_prototype reads the seg_<Label> logits the segmentation wrote), and
# the pial step's own solve supplies the thickness map further down.
DIRECT_OPTS=(--no-fsr --keep)
if [ ${SKIP_PIAL} -eq 0 ] || [ ${NO_CTH} -eq 1 ] ; then
	DIRECT_OPTS+=(--no-cth)
fi
"${SCRIPT_DIR}/scripts/dl+direct.sh" "${PASSTHROUGH[@]}" "${DIRECT_OPTS[@]}" "${T1}" "${DST}" \
	|| die "dl+direct.sh failed"

mkdir -p ${DST}/mri ${DST}/label ${DST}/surf || die "Could not create output subdirectories"

# Steps 2 and 3 are lines 26 and 36 of fast_surface_reconstruction.sh,
# unmodified. Everything between them there is FreeSurfer feeding
# mris_make_surfaces, which we do not run. Step 2 therefore skips the 256^3
# conform (which only mris_make_surfaces needed) and writes mri/ on the cropped
# grid, so the whole pipeline shares one grid and one tkrRAS frame; step 3 is
# unchanged and follows whatever grid mri/ is on. --conformed-space restores the
# 256^3 conform, whose surfaces the pial step then refuses (by design: their tkr
# frame differs from the reference grid's by half a voxel per odd crop dim).
python ${SCRIPT_DIR}/preparedata.py -inputpath ${DST} "${PREP_OPTS[@]}" || die "preparedata.py failed"
python ${SCRIPT_DIR}/dl_wm_surface_parallel_dev.py -inputpath ${DST} -outputpath ${DST} -ns 50 \
	|| die "dl_wm_surface_parallel_dev.py failed"

if [ ${SKIP_PIAL} -eq 0 ] ; then
	PIAL_OPTS=(--write-thickness)
	[ ${WM_FROM_SURFACE} -eq 0 ] && PIAL_OPTS+=(--seg-wm)
	[ ${SKIP_NAIVE} -eq 1 ] && PIAL_OPTS+=(--skip-naive)

	# Run as a module: field_pial_prototype uses relative imports (direct_cuda).
	# Point PYTHONPATH at the repo root so this works regardless of the caller's
	# working directory and without depending on an up-to-date installed copy.
	PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}" \
	python -m dldirect.field_pial_prototype \
		--prep-dir ${DST} --surf-dir ${DST}/surf --out-dir ${DST}/field_pial \
		--sigma ${SIGMA} --dip-threshold ${DIP_THRESHOLD} "${PIAL_OPTS[@]}" \
		|| die "field_pial_prototype failed"

	# The thickness map and seg above are on the cropped grid and in DiReCT.py's
	# layout, so these are the same two calls dl+direct.sh makes after DiReCT --
	# lifted here because --no-cth skipped them. MASK_VOLUME mirrors that
	# script's own choice: the hd-bet mask when --bet was used, else T1w_norm.
	MASK_VOLUME=${DST}/T1w_norm_noskull_mask.nii.gz
	[ -f "${MASK_VOLUME}" ] || MASK_VOLUME=${DST}/T1w_norm.nii.gz
	# The pial step leaves T1w_thickmap.nii.gz absent when its solve produced a
	# degenerate thickness map -- which is what a sub-voxel --sigma does, since
	# the gradient kernel then cannot propagate the field. That is a cortical-
	# thickness failure only: the surfaces above are still written, so treat the
	# missing map as "no thickness stats" rather than as a failed run.
	if [ -f "${DST}/T1w_thickmap.nii.gz" ] ; then
		python ${SCRIPT_DIR}/extract_stats.py "${DST}/T1w_thickmap.nii.gz" "${DST}/seg.nii.gz" \
			"${DST}/softmax_seg.nii.gz" "${SUBJECT_ID}" || die "extract_stats.py failed"
		python ${SCRIPT_DIR}/crop.py --revert 1 "${MASK_VOLUME}" "${DST}/T1w_thickmap.nii.gz" \
			"${DST}/T1w_norm_thickmap.nii.gz" || die "uncropping the thickness map failed"
	else
		echo "WARNING: no thickness map was produced (degenerate solve, e.g. --sigma below" >&2
		echo "         ~0.35); skipping thickness stats. Surfaces in ${DST}/surf are unaffected." >&2
	fi
fi

echo "Surfaces done (no FreeSurfer): ${DST}/surf"
