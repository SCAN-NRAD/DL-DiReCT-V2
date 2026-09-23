#!/usr/bin/env python
"""PROTOTYPE: topology correction on the GPU.

nighres' topology_correction (Bazin & Pham 2007) is the single most expensive
stage of the pial pipeline -- 17s of a 44s run, and it is a sequential
fast-marching front with a simple-point test at every step. This is an attempt
at the same idea in a form a GPU can execute.

The construction
----------------
A point p is SIMPLE for a set X when adding it (or removing it) changes no
topological invariant. For the (26, 6) connectivity pair the classical
characterisation (Malandain & Bertrand 1992) is entirely local to the 3x3x3
neighbourhood, and does not reference p's own value:

    C26 = number of 26-connected components of N26*(p) INTERSECT X          == 1
    C6  = number of  6-connected components of N18*(p) INTERSECT complement(X)
          that are 6-adjacent to p                                          == 1

Because the predicate ignores p itself, ONE test serves both deletion and
addition: p can be added to X iff the same two counts hold.

So: start from a seed that is genus 0 by construction (a single voxel), and
repeatedly add simple points of the mask. Every addition preserves topology, so
the result is genus 0 no matter where the growth stops. Where the mask has a
handle, the front meets itself and closing the loop is not simple, so the handle
is cut -- which is exactly the correction wanted.

Two things make it run on a GPU:

  * The predicate is a fixed 26-position problem. Connected components WITHIN
    the neighbourhood are found by label propagation against a constant 26x26
    adjacency matrix -- max-pooling over neighbours, a handful of iterations,
    no host involvement.
  * Adding several simple points at once can break topology if they touch, so
    candidates are partitioned into 8 subfields by coordinate parity. Within a
    subfield every pair is at Chebyshev distance >= 2, hence not 26-adjacent,
    which is the condition the subfield-parallel thinning results rely on.

Candidates are only ever the front -- the mask voxels 6-adjacent to the current
set -- so neighbourhoods are gathered for ~1e4-1e5 voxels, not for the volume.

STATUS: prototype. The topology of the OUTPUT is verified empirically (Euler
characteristic of the marching-cubes surface), not asserted from the theory --
see topology_gpu_check in the test script. It is not wired into the pipeline.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# the fixed 3x3x3 tables
# ---------------------------------------------------------------------------
def neighbour_tables(device):
    """offsets (26,3) and the adjacency/membership masks the predicate needs."""
    off = np.array([(dz, dy, dx)
                    for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dz, dy, dx) != (0, 0, 0)], dtype=np.int64)
    d = np.abs(off[:, None, :] - off[None, :, :])
    a26 = (d.max(-1) == 1)                     # 26-adjacent within the box
    a6 = (d.sum(-1) == 1)                      # 6-adjacent within the box
    l1 = np.abs(off).sum(1)
    is18 = l1 <= 2                             # N18*: drop the 8 corners
    is6 = l1 == 1                              # the 6 face neighbours
    t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dt)
    return dict(off=t(off, torch.int64), a26=t(a26, torch.bool), a6=t(a6, torch.bool),
                is18=t(is18, torch.bool), is6=t(is6, torch.bool))


def _components(occ, adj, iters=12):
    """Label the connected components of each row's occupied positions.

    occ [N, 26] bool, adj [26, 26] bool. Returns labels [N, 26] int16 where a
    component carries the largest position index it contains (so the number of
    components is the count of k with label[k] == k), and -1 where unoccupied.
    """
    N, K = occ.shape
    ar = torch.arange(K, device=occ.device, dtype=torch.int16)
    lab = torch.where(occ, ar.expand(N, K), torch.full_like(ar.expand(N, K), -1))
    link = adj.unsqueeze(0) & occ.unsqueeze(1) & occ.unsqueeze(2)   # [N,K,K] both ends set
    for _ in range(iters):
        cand = torch.where(link, lab.unsqueeze(1).expand(N, K, K),
                           torch.full((1, 1, 1), -1, dtype=torch.int16, device=occ.device))
        new = torch.maximum(lab, cand.amax(dim=2))
        if torch.equal(new, lab):
            break
        lab = new
    return lab


def _n_components(occ, adj, T, comp_iters, face_only):
    """Components of `occ` under `adj`; if face_only, count only those containing
    one of the 6 face neighbours (i.e. those adjacent to p itself)."""
    N = occ.shape[0]
    lab = _components(occ, adj, comp_iters)
    if not face_only:
        ar = torch.arange(26, device=occ.device, dtype=torch.int16)
        return ((lab == ar) & occ).sum(1)
    face = occ & T['is6']
    slot = torch.where(face, lab.long(), torch.full_like(lab.long(), 26))
    present = torch.zeros(N, 27, dtype=torch.bool, device=occ.device)
    present.scatter_(1, slot, torch.ones_like(slot, dtype=torch.bool))
    return present[:, :26].sum(1)


def is_simple(occ, T, comp_iters=12, pair='26-6'):
    """Simple-point predicate for rows of neighbourhood occupancy.

    occ [N, 26] bool: is each of p's 26 neighbours in X? p itself is not
    represented, which is why one predicate covers addition and deletion.

    pair='26-6'  object 26-connected, background 6-connected
    pair='6-26'  the dual: object 6-connected, background 26-connected. This is
                 the convention the shipped pipeline uses (nighres is called
                 with connectivity '6/18'), and it is the one whose boundary a
                 marching-cubes surface can represent.
    """
    if pair == '26-6':
        n_obj = _n_components(occ, T['a26'], T, comp_iters, face_only=False)
        n_bg = _n_components((~occ) & T['is18'], T['a6'], T, comp_iters, face_only=True)
    elif pair == '6-26':
        n_obj = _n_components(occ & T['is18'], T['a6'], T, comp_iters, face_only=True)
        n_bg = _n_components(~occ, T['a26'], T, comp_iters, face_only=False)
    else:
        raise ValueError('pair must be "26-6" or "6-26", got %r' % (pair,))
    return (n_obj == 1) & (n_bg == 1)


# ---------------------------------------------------------------------------
# the growth
# ---------------------------------------------------------------------------
def _face_dilate(x):
    """6-connected dilation of a padded [1,1,D,H,W] float volume."""
    y = x.clone()
    for dim in (2, 3, 4):
        y = torch.maximum(y, x.roll(1, dims=dim))
        y = torch.maximum(y, x.roll(-1, dims=dim))
    return y


def correct_topology(mask, device=None, seed=None, max_rounds=400, verbose=True,
                     comp_iters=12, pair='26-6', priority=None, n_bands=8,
                     protect_above=None, protect_mask=None):
    """Genus-0 subset of `mask`, grown by adding simple points only.

    mask      [D,H,W] boolean/int array
    seed      optional [D,H,W] boolean start set; must itself be genus 0.
              Defaults to the single deepest voxel of the mask, which trivially
              is.
    priority  optional [D,H,W] float, higher = admit sooner. Without it the
              front advances by distance, so WHERE a handle gets cut is decided
              by traversal geometry -- arbitrary. The segmentation this runs on
              is a hard argmax of a soft model output, and on the fill rim 25.8%
              of voxels sit at P in [0.4, 0.6]; the disputed spur that made the
              GPU and nighres corrections disagree on one measured hemisphere
              had P(WM) = 0.514 with 0.501 and 0.507 beside it. Growing in
              descending confidence closes the confident tissue first, so the
              cut is forced onto the least confident voxels in the loop.
    n_bands   number of descending priority bands; each is grown to exhaustion
              before the next is admitted. Ignored when priority is None.
    protect_above
              a priority floor. Tissue at or above it is never sacrificed: once
              the topology-preserving growth has converged, any remaining mask
              voxel above the floor that touches the current set is added
              REGARDLESS of whether it is simple. The result is then no longer
              guaranteed genus 0 -- the defect is accepted instead of being paid
              for in confident tissue. Without it the growth's only currency is
              tissue, so a mask with many defects necessarily loses some.
              Needs `priority`; the floor is in that array's units.
    protect_mask
              a boolean volume of tissue that must never be sacrificed,
              applied the same way as protect_above but by location rather than
              confidence. Use it to say WHERE a cut is allowed to be: the
              correction has no notion of CSF, it only finds where the shape is
              topologically wrong, so without a locality constraint it will cut
              at the white-matter interface and at the outer brain margin as
              readily as in a sulcus. Must be the same shape as `mask`, i.e.
              padded the same way `priority` is.

    Returns (corrected [D,H,W] bool, info dict).
    """
    device = torch.device(device) if device is not None else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = neighbour_tables(device)

    m = torch.from_numpy(np.ascontiguousarray(np.asarray(mask) > 0)).to(device)
    D, H, W = m.shape
    # one voxel of padding so a neighbourhood never leaves the array
    M = torch.zeros((D + 2, H + 2, W + 2), dtype=torch.bool, device=device)
    M[1:-1, 1:-1, 1:-1] = m
    Dp, Hp, Wp = M.shape
    stride = torch.tensor([Hp * Wp, Wp, 1], device=device, dtype=torch.int64)
    off_flat = (T['off'] * stride).sum(1)                   # [26] flat offsets

    X = torch.zeros_like(M)
    if seed is None:
        from scipy.ndimage import distance_transform_edt
        edt = distance_transform_edt(np.asarray(mask) > 0)
        z, y, x = np.unravel_index(int(np.argmax(edt)), edt.shape)
        X[z + 1, y + 1, x + 1] = True
        if verbose:
            print('  seed: single voxel at (%d,%d,%d), depth %.2f' % (z, y, x, edt.max()))
    else:
        X[1:-1, 1:-1, 1:-1] = torch.from_numpy(np.asarray(seed) > 0).to(device)

    # subfield membership by coordinate parity, in PADDED coordinates
    gz, gy, gx = torch.meshgrid(torch.arange(Dp, device=device),
                                torch.arange(Hp, device=device),
                                torch.arange(Wp, device=device), indexing='ij')
    parity = ((gz % 2) * 4 + (gy % 2) * 2 + (gx % 2)).reshape(-1)

    Xf = X.reshape(-1)
    Mf = M.reshape(-1)

    if priority is None:
        bands = [None]
        Pf = None
    else:
        P = torch.zeros_like(M, dtype=torch.float32)
        P[1:-1, 1:-1, 1:-1] = torch.as_tensor(np.asarray(priority, np.float32)).to(device)
        Pf = P.reshape(-1)
        vals = Pf[Mf]
        # descending band edges, by quantile so each band holds comparable mass
        qs = torch.linspace(1.0, 0.0, n_bands + 1, device=device)[1:]
        bands = [float(torch.quantile(vals, q)) for q in qs]
        bands[-1] = float(vals.min()) - 1.0            # last band admits everything
        if verbose:
            print('  priority bands: %s' % ' '.join('%.3f' % b for b in bands))

    added_total = 0
    rounds = 0
    for thr in bands:
        allowed = Mf if thr is None else (Mf & (Pf >= thr))
        for rnd in range(max_rounds):
            rounds += 1
            added_round = 0
            for sub in range(8):
                front = _face_dilate(Xf.reshape(1, 1, Dp, Hp, Wp).float()).reshape(-1) > 0
                cand = (allowed & ~Xf & front & (parity == sub)).nonzero(as_tuple=True)[0]
                if cand.numel() == 0:
                    continue
                occ = Xf[cand.unsqueeze(1) + off_flat.unsqueeze(0)]
                ok = is_simple(occ, T, comp_iters, pair=pair)
                sel = cand[ok]
                if sel.numel():
                    Xf[sel] = True
                    added_round += int(sel.numel())
            added_total += added_round
            if verbose and rounds % 20 == 0:
                print('  round %3d: %8d voxels (+%d this round)'
                      % (rounds, int(Xf.sum()), added_round))
            if added_round == 0:
                break

    protected = 0
    if protect_above is not None or protect_mask is not None:
        # start with NOTHING protected and OR in each constraint; initialising
        # to Mf protects the whole mask and silently disables the correction
        keep = torch.zeros_like(Mf)
        if protect_above is not None:
            if Pf is None:
                raise ValueError('protect_above needs priority=')
            keep = keep | (Mf & (Pf >= float(protect_above)))
        if protect_mask is not None:
            pm = torch.zeros_like(M)
            pm[1:-1, 1:-1, 1:-1] = torch.from_numpy(
                np.asarray(protect_mask, bool)).to(device)
            keep = keep | (Mf & pm.reshape(-1))
        for _ in range(max_rounds):
            front = _face_dilate(Xf.reshape(1, 1, Dp, Hp, Wp).float()).reshape(-1) > 0
            cand = keep & ~Xf & front
            k = int(cand.sum())
            if k == 0:
                break
            Xf[cand] = True          # unconditional: topology is not preserved here
            protected += k
        if verbose:
            print('  protected %d voxels at or above %.4f' % (protected, protect_above))

    out = Xf.reshape(Dp, Hp, Wp)[1:-1, 1:-1, 1:-1].cpu().numpy()
    info = dict(rounds=rounds, added=added_total, protected=protected,
                filled=int(out.sum()),
                mask=int(m.sum().item()),
                coverage=float(out.sum()) / max(int(m.sum().item()), 1))
    if verbose:
        print('  %d rounds, %d/%d mask voxels retained (%.2f%%)'
              % (info['rounds'], info['filled'], info['mask'], 100 * info['coverage']))
    return out, info
