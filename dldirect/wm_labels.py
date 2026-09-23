"""The label set filled into the white-matter interior, in one place.

`preparedata.py` (filled.mgz) and `dl_wm_surface_parallel_dev.py` (the mask
actually tessellated into ?h.white) must agree on this set. They used to hold
two copies of the same list expression and drifted apart: filled.mgz dropped
the hippocampus while the tessellated mask kept it, so the surface enclosed a
structure the pipeline believed it had excluded. This module is the single
definition, plus a record written next to the run so the second script cannot
be given different options from the first.

Everything not excluded is deliberately filled: the subcortical structures,
ventricles and ventral DC are what make the hemisphere mask simply connected
for topology correction. Excluding one removes it from the white surface's
interior, which is a real change to the surface, not a cosmetic one.
"""
import os

# Excluded for every run. Cerebellum is not cerebrum; the hippocampus is
# archicortex with no cortical ribbon over it, and enclosing it put the white
# surface around a structure the pial then had to be pinned to (see
# build_no_push_mask in field_pial_prototype).
ALWAYS_EXCLUDED = ('Cerebellum', 'Hippocampus')

# Excludable on request. The amygdala is the other allocortical structure the
# hemisphere fill swallows -- measured on bert, 99.1% of it lies inside the
# white surface, against 1.1% of the hippocampus.
OPTIONAL = ('Amygdala',)

RECORD = 'wm_fill_exclusions.txt'


def excluded_structures(extra=()):
    """Structure suffixes dropped from the hemisphere fill."""
    return tuple(ALWAYS_EXCLUDED) + tuple(extra)


def hemisphere_labels(df_labels, side, excluded):
    """Label IDs filled for one hemisphere. `side` is 'Left' or 'Right'."""
    skip = {'%s-%s' % (side, s) for s in excluded}
    return [df_labels['ID'][x] for x in df_labels['ID'].keys()
            if x.startswith(side) and x not in skip]


def ribbon_labels(df_labels, region, excluded):
    """Label IDs of one hemisphere's cortical RIBBON: the WM fill plus the
    cortex parcels.

    The fill's labels are named 'Left-*' / 'Right-*'; a parcellated aseg names
    the cortex per gyrus as 'lh-*' / 'rh-*'. Both families are needed -- the
    fill alone is white matter, not the ribbon.
    """
    side = 'Left' if region == 'lh' else 'Right'
    skip = {'%s-%s' % (side, s) for s in excluded}
    ids = list(hemisphere_labels(df_labels, side, excluded))
    ids += [i for n, i in df_labels['ID'].items()
            if n.startswith(region + '-') and n not in skip]
    return sorted(set(ids))


def write_record(mri_dir, excluded):
    """Record the exclusions beside the run, so the surface step inherits them."""
    with open(os.path.join(mri_dir, RECORD), 'w') as f:
        f.write('\n'.join(excluded) + '\n')


def read_record(mri_dir):
    """Exclusions recorded by preparedata.py; the always-excluded set if absent
    (a run that predates the record, which by definition used only those)."""
    path = os.path.join(mri_dir, RECORD)
    if not os.path.exists(path):
        return tuple(ALWAYS_EXCLUDED)
    with open(path) as f:
        return tuple(x.strip() for x in f if x.strip())
