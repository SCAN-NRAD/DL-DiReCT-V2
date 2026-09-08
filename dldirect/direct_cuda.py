"""
GPU-accelerated DiReCT (KellyKapowski) cortical thickness estimation in PyTorch.

Reimplements the ANTs itkDiReCTImageFilter algorithm using PyTorch tensors
and F.grid_sample for GPU acceleration.

Reference: Das SR, Avants BB, Grossman M, Gee JC. Registration based cortical
thickness measurement. Neuroimage. 2009;45(3):867-879.
"""

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import time
import os
import sys


def get_device(device=None):
    """Auto-detect best available device (cuda > mps > cpu)."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _gaussian_kernel_1d(sigma, device, truncate=4.0):
    """Create a 1D Gaussian kernel."""
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        radius = 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_smooth_3d(vol, sigma, device, zero_boundary=True, truncate=4.0):
    """Separable 3D Gaussian smoothing using three 1D convolutions.

    Args:
        vol: [1, C, D, H, W] tensor
        sigma: smoothing sigma in voxels — scalar for isotropic,
               or (sigma_d, sigma_h, sigma_w) tuple for anisotropic
        zero_boundary: if True, zero out boundary voxels after smoothing
    """
    if isinstance(sigma, (int, float)):
        sigma_d = sigma_h = sigma_w = float(sigma)
    else:
        sigma_d, sigma_h, sigma_w = sigma

    if max(sigma_d, sigma_h, sigma_w) <= 0:
        return vol

    C = vol.shape[1]

    # Smooth along D axis
    if sigma_d > 0:
        kernel = _gaussian_kernel_1d(sigma_d, device, truncate)
        k = kernel.numel()
        pad = k // 2
        kd = kernel.reshape(1, 1, k, 1, 1).expand(C, -1, -1, -1, -1)
        vol = F.conv3d(F.pad(vol, (0, 0, 0, 0, pad, pad), mode='replicate'),
                       kd, groups=C)

    # Smooth along H axis
    if sigma_h > 0:
        kernel = _gaussian_kernel_1d(sigma_h, device, truncate)
        k = kernel.numel()
        pad = k // 2
        kh = kernel.reshape(1, 1, 1, k, 1).expand(C, -1, -1, -1, -1)
        vol = F.conv3d(F.pad(vol, (0, 0, pad, pad, 0, 0), mode='replicate'),
                       kh, groups=C)

    # Smooth along W axis
    if sigma_w > 0:
        kernel = _gaussian_kernel_1d(sigma_w, device, truncate)
        k = kernel.numel()
        pad = k // 2
        kw = kernel.reshape(1, 1, 1, 1, k).expand(C, -1, -1, -1, -1)
        vol = F.conv3d(F.pad(vol, (pad, pad, 0, 0, 0, 0), mode='replicate'),
                       kw, groups=C)

    if zero_boundary:
        vol[:, :, 0, :, :] = 0
        vol[:, :, -1, :, :] = 0
        vol[:, :, :, 0, :] = 0
        vol[:, :, :, -1, :] = 0
        vol[:, :, :, :, 0] = 0
        vol[:, :, :, :, -1] = 0

    return vol


def _gaussian_deriv_kernel_1d(sigma, device, truncate=4.0):
    """Create a 1D Gaussian derivative kernel from the normalized smoothing kernel.

    This is the analytical derivative of the normalized Gaussian:
        d/dx [G(x)] = -(x / sigma^2) * G(x)
    where G(x) is the normalized Gaussian (sum=1).
    """
    smooth = _gaussian_kernel_1d(sigma, device, truncate)
    radius = len(smooth) // 2
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    # Note: positive sign because F.conv3d computes cross-correlation (not
    # convolution), which flips the kernel. The analytical derivative of the
    # Gaussian is -(x/sigma^2)*G(x), but cross-correlation with that kernel
    # gives the negated derivative. Using (x/sigma^2)*G(x) corrects for this.
    return (x / (sigma * sigma)) * smooth


def gaussian_gradient_3d(vol, sigma, device, voxel_size=None):
    """Gradient via separable Gaussian derivative convolution.

    For each axis, convolves with the derivative kernel along that axis and
    the smoothing kernel along the other two. Equivalent to ANTs'
    GradientRecursiveGaussianImageFilter but as FIR filters.

    Args:
        vol: [1, 1, D, H, W] tensor
        sigma: smoothing sigma in voxels — scalar for isotropic,
               or (sigma_d, sigma_h, sigma_w) tuple for anisotropic
        voxel_size: (vd, vh, vw) in mm, used to scale gradient to physical
                    units. None = isotropic (no scaling).
    Returns:
        [1, 3, D, H, W] gradient (d/dD, d/dH, d/dW)
    """
    if isinstance(sigma, (int, float)):
        sigmas = (float(sigma), float(sigma), float(sigma))
    else:
        sigmas = tuple(sigma)

    C = vol.shape[1]

    # Build per-axis smooth and derivative kernels
    smooth_kernels = [_gaussian_kernel_1d(s, device) for s in sigmas]
    deriv_kernels = [_gaussian_deriv_kernel_1d(s, device) for s in sigmas]

    grads = []
    for axis in range(3):  # D, H, W
        tmp = vol
        for ax in range(3):
            if ax == axis:
                k = deriv_kernels[ax]
            else:
                k = smooth_kernels[ax]
            klen = k.numel()
            pad = klen // 2
            if ax == 0:  # D
                k_shaped = k.reshape(1, 1, klen, 1, 1).expand(C, -1, -1, -1, -1)
                tmp = F.conv3d(F.pad(tmp, (0, 0, 0, 0, pad, pad), mode='replicate'),
                               k_shaped, groups=C)
            elif ax == 1:  # H
                k_shaped = k.reshape(1, 1, 1, klen, 1).expand(C, -1, -1, -1, -1)
                tmp = F.conv3d(F.pad(tmp, (0, 0, pad, pad, 0, 0), mode='replicate'),
                               k_shaped, groups=C)
            else:  # W
                k_shaped = k.reshape(1, 1, 1, 1, klen).expand(C, -1, -1, -1, -1)
                tmp = F.conv3d(F.pad(tmp, (pad, pad, 0, 0, 0, 0), mode='replicate'),
                               k_shaped, groups=C)
        grads.append(tmp)

    result = torch.cat(grads, dim=1)

    # Scale gradient components from per-voxel to per-mm if anisotropic
    if voxel_size is not None:
        vd, vh, vw = voxel_size
        result[:, 0] /= vd
        result[:, 1] /= vh
        result[:, 2] /= vw

    return result



def direction_masked_smooth_3d(vol, sigma, device, truncate=2.0, mode='hard'):
    """Gaussian smoothing of a vector field that refuses to average across a
    direction reversal.

    A neighbour only contributes to a voxel if its vector has a non-negative dot
    product with that voxel's own vector. Across a thin gyral blade the two banks
    carry near-antiparallel velocities, and an isotropic Gaussian blends them into
    a common translation; this keeps the two sides separate. Where the centre
    vector is essentially zero there is no direction to preserve, so the filter
    falls back to the ordinary Gaussian and stagnant regions can still be filled.
    """
    r = max(1, int(truncate * sigma + 0.5))
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()

    D, H, W = vol.shape[2:]
    padded = F.pad(vol, (r, r, r, r, r, r), mode='replicate')
    acc = torch.zeros_like(vol)
    wacc = torch.zeros((vol.shape[0], 1, D, H, W), device=device)
    centre_zero = (vol ** 2).sum(dim=1, keepdim=True).sqrt() < 1e-6

    norm_c = (vol ** 2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6)
    for dz in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                wgt = float(g[dz + r] * g[dy + r] * g[dx + r])
                if wgt < 1e-6:
                    continue
                sh = padded[:, :, r + dz:r + dz + D, r + dy:r + dy + H, r + dx:r + dx + W]
                dot = (sh * vol).sum(dim=1, keepdim=True)
                if mode == 'hard':
                    dw = (dot > 0).to(vol.dtype)
                else:
                    cos = dot / (norm_c * (sh ** 2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6))
                    # 'relu' drops anything past 90 degrees, 'soft' only attenuates
                    dw = cos.clamp(min=0.0) if mode == 'relu' else (1.0 + cos) * 0.5
                dw = torch.where(centre_zero, torch.ones_like(dw), dw)
                acc = acc + wgt * dw * sh
                wacc = wacc + wgt * dw
    return torch.where(wacc > 1e-6, acc / wacc.clamp(min=1e-6), vol)



def selective_masked_smooth_3d(vol, sigma, device, truncate=2.0, mode='hard',
                               coherence_threshold=0.5, gate='coherence',
                               return_stats=False):
    """Plain Gaussian smoothing everywhere except where the field is locally bipolar.

    Coherence is |G * v| / (G * |v|): near 1 where the neighbourhood points one
    way, near 0 where opposing vectors cancel — which is what a thin gyral blade
    looks like, the two banks carrying near-antiparallel velocity. Only those
    voxels get the direction-masked filter, so the Gaussian keeps regularising
    the rest of the volume.
    """
    plain = gaussian_smooth_3d(vol, sigma, device, zero_boundary=False, truncate=truncate)
    mag = (vol ** 2).sum(dim=1, keepdim=True).sqrt()
    smooth_mag = gaussian_smooth_3d(mag, sigma, device, zero_boundary=False, truncate=truncate)
    coherence = (plain ** 2).sum(dim=1, keepdim=True).sqrt() / smooth_mag.clamp(min=1e-6)

    if gate == 'direction':
        # the smoothed vector has turned away from the voxel's own vector, i.e.
        # the neighbourhood has overruled it — that is the damage we care about
        cos_turn = (plain * vol).sum(dim=1, keepdim=True) / (
            (plain ** 2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-6) * mag.clamp(min=1e-6))
        bipolar = (cos_turn < coherence_threshold) & (mag > 1e-6)
    else:
        bipolar = (coherence < coherence_threshold) & (smooth_mag > 1e-6)
    if not bool(bipolar.any()):
        return (plain, 0.0) if return_stats else plain

    masked = direction_masked_smooth_3d(vol, sigma, device, truncate=truncate, mode=mode)
    out = torch.where(bipolar, masked, plain)
    if return_stats:
        return out, float(bipolar.float().mean())
    return out


def _make_identity_grid(shape, device):
    """Create identity sampling grid for F.grid_sample in (x=W, y=H, z=D) order."""
    D, H, W = shape
    gd, gh, gw = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij')
    return torch.stack([gw, gh, gd], dim=-1).unsqueeze(0)


def warp_image(image, disp_field, identity_grid):
    """Warp image by displacement field.

    Args:
        image: [1, C, D, H, W]
        disp_field: [1, 3, D, H, W] displacement in voxels (d, h, w order)
        identity_grid: [1, D, H, W, 3] precomputed identity grid
    """
    D, H, W = image.shape[2:]
    # Convert voxel displacement (d,h,w) to normalized grid coords (x=w, y=h, z=d)
    norm_disp = torch.stack([
        disp_field[:, 2] * 2.0 / (W - 1),
        disp_field[:, 1] * 2.0 / (H - 1),
        disp_field[:, 0] * 2.0 / (D - 1),
    ], dim=-1)
    grid = identity_grid + norm_disp
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True)


def compose_fields(A, B, identity_grid):
    """Compose displacement fields: result(x) = A(x + B(x)) + B(x)."""
    A_warped = warp_image(A, B, identity_grid)
    return A_warped + B


def invert_field(field, identity_grid, max_iter=20, initial=None):
    """Invert displacement field via damped fixed-point iteration.

    Matches ANTs itkInvertDisplacementFieldImageFilter:
    - Proportional clamping: per-voxel updates are scaled down when their
      norm exceeds epsilon * maxErrorNorm, preventing outlier-driven overshoot.
    - Early stopping: converges when max residual <= 0.1 or mean <= 0.001.
    - Epsilon: 0.75 first iteration, 0.5 thereafter.

    Args:
        field: [1, 3, D, H, W] displacement field to invert
        identity_grid: precomputed identity grid
        max_iter: number of iterations (default 20, warm-started)
        initial: [1, 3, D, H, W] initial estimate (None = zeros)
    """
    inv = initial.clone() if initial is not None else torch.zeros_like(field)
    for i in range(max_iter):
        residual = warp_image(field, inv, identity_grid) + inv

        # Per-voxel norm for proportional clamping and convergence check
        scaled_norm = (residual * residual).sum(dim=1, keepdim=True).sqrt()
        max_error = scaled_norm.max()

        # Early stopping (ANTs thresholds: max <= 0.1, mean <= 0.001)
        if max_error <= 0.1 or scaled_norm.mean() <= 0.001:
            break

        epsilon = 0.75 if i == 0 else 0.5
        threshold = epsilon * max_error

        # Proportional clamping: scale down outlier voxels
        clamp_scale = torch.where(scaled_norm > threshold,
                                  threshold / scaled_norm.clamp(min=1e-10),
                                  torch.ones_like(scaled_norm))

        inv = inv - epsilon * clamp_scale * residual
    return inv


def extract_wm_contours(seg_tensor):
    """Extract WM voxels adjacent to GM (WM/GM interface).

    Args:
        seg_tensor: [1, 1, D, H, W] segmentation (2=GM, 3=WM)
    Returns:
        [1, 1, D, H, W] binary mask of WM contour voxels
    """
    gm_mask = (seg_tensor == 2).float()
    gm_dilated = F.max_pool3d(gm_mask, kernel_size=3, stride=1, padding=1)
    wm_mask = (seg_tensor == 3).float()
    return wm_mask * gm_dilated



@torch.no_grad()

def _save_velocity_fields(prefix, ref_img, inverse_snapshots, forward_snapshots):
    """Write the fields in the layout the ANTs backend produces.

    ANTs writes <prefix>ForwardVelocityField.nii.gz / <prefix>InverseVelocityField.nii.gz
    as [D, H, W, T, 3] float32, with the components in (d, h, w) array order,
    which is the order the CUDA fields already use. Verified against the ANTs
    output on a test subject: component-wise r = 0.998 with no permutation.

    Two differences from ANTs remain and are deliberately not papered over:
    ANTs leaves its first two time points at zero and its sequence corresponds
    to CUDA's shifted by two (ANTs[t] ~ 1.08 * CUDA[t-2]), so index t does not
    denote the same point of the integration in both backends.
    """
    assert ref_img is not None, "ref_img required when velocity_field_prefix is set"

    for name, snaps in [('Forward', forward_snapshots), ('Inverse', inverse_snapshots)]:
        # [3, D, H, W, T] -> [D, H, W, T, 3]
        arr = np.stack(snaps, axis=-1).transpose(1, 2, 3, 4, 0).astype(np.float32)
        img = nib.Nifti1Image(arr, ref_img.affine)
        img.header['xyzt_units'] = 10
        nib.save(img, '{}{}VelocityField.nii.gz'.format(prefix, name))


def kelly_kapowski_cuda(
    seg, gm_prob, wm_prob,
    max_iterations=45,
    gradient_step=0.025,
    smoothing_sigma=1.0,
    gradient_sigma=None,
    velocity_smooth_sigma=1.2247,
    velocity_smooth_sigma_start=None,
    velocity_smooth_sigma_power=1.0,
    velocity_smooth_masked=False,
    velocity_smooth_mask_mode='hard',
    velocity_smooth_selective=None,
    velocity_smooth_gate='coherence',
    num_integration_points=10,
    thickness_prior=10.0,
    max_invert_iterations=20,
    device=None,
    verbose=True,
    save_fields_dir=None,
    velocity_field_prefix=None,
    cumulative_fields=True,
    return_velocity=False,
    gradient_gate=1e-3,
    ref_img=None,
    voxel_size=None,
):
    """GPU-accelerated KellyKapowski cortical thickness estimation.

    Faithfully reimplements the ANTs itkDiReCTImageFilter algorithm.

    Args:
        seg: [D, H, W] uint8 numpy array, 0=background, 2=GM, 3=WM
        gm_prob: [D, H, W] float32 numpy array, gray matter probability
        wm_prob: [D, H, W] float32 numpy array, white matter probability
        max_iterations: outer loop iterations (default 45)
        gradient_step: Euler step size in mm (default 0.025)
        smoothing_sigma: sigma for gradient smoothing (mm) and hit/total
            smoothing (voxels). Same value, different units per context —
            matches ANTs m_SmoothingVariance inconsistency. Default 1.0.
        velocity_smooth_sigma_start: if set, the velocity smoothing sigma is
            annealed geometrically from this value at the first iteration down
            to velocity_smooth_sigma at the last. Wider early smoothing diffuses
            velocity into regions where the field is otherwise stagnant.
        velocity_smooth_sigma: sigma for velocity field smoothing in voxels
            (default sqrt(1.5) ≈ 1.2247, matching ANTs m_SmoothingVelocityFieldVariance=1.5)
        num_integration_points: inner integration steps (default 10)
        thickness_prior: maximum expected thickness in mm (default 10.0)
        max_invert_iterations: iterations for field inversion (default 20)
        device: torch device (auto-detected if None)
        verbose: print progress
        velocity_field_prefix: if set, write the final iteration's fields as
            <prefix>ForwardVelocityField.nii.gz and <prefix>InverseVelocityField.nii.gz
            in the same layout as the ANTs backend, i.e. [D, H, W, T, 3] float32.
        save_fields_dir: if set, save displacement fields as NIfTI to this directory.
            Saves per-iteration 4D volumes (3 components × integration points) for both
            the inverse field and forward (integrated) field.
        gradient_gate: voxels whose gradient magnitude is at or below this are
            given no direction and cannot move. NOTE this is an ABSOLUTE
            threshold while the derivative kernel's amplitude varies strongly
            with smoothing_sigma: measured on the development subject, the
            median GM gradient spans ~2000x from sigma=0.35 to sigma=1.0, and
            the fraction of GM voxels falling under the gate goes 3.7% (sigma
            1.0) -> 15.7% (0.75) -> 39% (0.5) -> 54% (0.35) -> 100% (0.2). So
            the same constant means very different things at different sigma.
        return_velocity: return (thickness, velocity) instead of thickness, with
            the velocity array as written to Velocity.nii.gz (None if no
            velocity_field_prefix was given).
        cumulative_fields: also write <prefix>Forward/InverseVelocityField.nii.gz.
            These need a [3,D,H,W] GPU->CPU copy per integration point and cost
            far more than the solve itself. Velocity.nii.gz -- the underlying
            field, and the only one the pial propagation reads -- is written
            either way, so set False when the cumulative fields are not needed.
        ref_img: nibabel image for affine/header when saving fields (required if save_fields_dir is set)
        voxel_size: (vd, vh, vw) voxel size in mm. None = (1, 1, 1) isotropic.
            When set, smoothing_sigma is converted from mm to per-axis voxel
            sigmas for gradient computation only. Hit/total and velocity
            smoothing use their sigma values directly in voxel units (matching
            ANTs SetUseImageSpacing(false)). Thickness is returned in mm.

    Returns:
        [D, H, W] float32 numpy array of cortical thickness in mm
    """
    device = get_device(device)
    if verbose:
        print(f"DiReCT CUDA: using device {device}")

    shape = seg.shape
    D, H, W = shape

    # Voxel size handling for anisotropic voxels.
    # All displacement fields remain in VOXEL units throughout. Voxel size
    # is used for: (a) per-axis gradient sigma, (b) physical-space gradient
    # normalization, (c) converting voxel displacement to mm for thickness.
    #
    # ANTs uses inconsistent units for smoothing parameters (see
    # docs/ants_smoothing_units.md):
    # - Gradient: sigma in mm (physical), filter converts internally
    # - Hit/total smoothing: variance in voxels², UseImageSpacing(false)
    # - Velocity smoothing: variance in voxels², no spacing conversion
    #
    # Our API takes sigma (not variance) for all parameters, with defaults
    # smoothing_sigma=1.0, velocity_smooth_sigma=1.2247 (=sqrt(1.5)).
    # Only the gradient sigma needs mm→voxel conversion; the smoothing
    # sigmas are already in voxel units and must NOT be divided by voxel_size.
    # `smoothing_sigma` normally sets BOTH the gradient (differentiation) sigma
    # and the hit/total accumulation sigma. `gradient_sigma`, when given,
    # overrides only the former, so the differentiation scale can be varied
    # independently of the accumulation smoothing.
    grad_sigma_mm = smoothing_sigma if gradient_sigma is None else gradient_sigma
    if voxel_size is None:
        voxel_size_t = None
        grad_sigma_vox = grad_sigma_mm          # scalar, backward-compatible
        smooth_sigma_vox = smoothing_sigma
        vel_sigma_vox = velocity_smooth_sigma
    else:
        vd, vh, vw = voxel_size
        voxel_size_t = torch.tensor([vd, vh, vw], device=device, dtype=torch.float32).reshape(1, 3, 1, 1, 1)
        # Gradient: sigma in mm → voxel sigma using the in-plane resolution.
        # We use the mode of voxel sizes (after rounding to 2 significant
        # figures) as the in-plane resolution. This avoids per-axis sigma
        # that becomes sub-voxel on coarse axes (e.g., 1.0/5.0 = 0.2 vox),
        # while correctly scaling for the resolution where cortical detail
        # actually exists. For typical anisotropic MRI, 2 of 3 axes share
        # the in-plane resolution, so the mode picks it up robustly.
        def _round_sig(x, sig=2):
            """Round to sig significant figures."""
            if x == 0:
                return 0.0
            from math import log10, floor
            return round(x, sig - 1 - int(floor(log10(abs(x)))))
        rounded = [_round_sig(v) for v in (vd, vh, vw)]
        # Mode: most frequent value; fall back to median if all differ
        from collections import Counter
        counts = Counter(rounded)
        mode_val = counts.most_common(1)[0][0]
        if counts.most_common(1)[0][1] == 1:
            # All three differ — use median
            mode_val = sorted(rounded)[1]
        grad_sigma_vox = grad_sigma_mm / mode_val
        # Hit/total and velocity smoothing: sigma in voxels, same on all axes
        # (ANTs uses SetUseImageSpacing(false) — no spacing conversion)
        smooth_sigma_vox = smoothing_sigma
        vel_sigma_vox = velocity_smooth_sigma
        if verbose:
            print(f"  voxel_size={voxel_size}, in-plane={mode_val:.3f}mm, "
                  f"grad_sigma_vox={grad_sigma_vox:.4f}, "
                  f"smooth_sigma_vox={smooth_sigma_vox}, vel_sigma_vox={vel_sigma_vox}")

    # A sub-voxel gradient sigma cannot be represented on an integer grid: the
    # sampled smoothing kernel degenerates towards [0, 1, 0] and its analytic
    # derivative towards zero, so the gradient falls under the (grad_mag > 1e-3)
    # gate below and NOTHING propagates -- the solve then returns an all-zero
    # thickness map. That is a property of the sampling, not a bug to clamp
    # away, so warn loudly rather than silently returning zeros.
    _dk = _gaussian_deriv_kernel_1d(grad_sigma_vox, device).abs().max().item()
    if _dk <= 1e-3:
        print(f"WARNING: grad_sigma_vox={grad_sigma_vox:.4f} is too small to "
              f"differentiate on this grid (max|deriv kernel|={_dk:.2e} <= the "
              f"1e-3 gradient gate). The velocity field will not propagate and "
              f"the thickness map will be all zeros. Cortical thickness from "
              f"this solve is unusable; surfaces may still be fine.",
              file=sys.stderr)

    # Move inputs to device as [1, 1, D, H, W]
    seg_t = torch.from_numpy(seg.astype(np.float32)).to(device).reshape(1, 1, D, H, W)
    gm_prob_t = torch.from_numpy(gm_prob.astype(np.float32)).to(device).reshape(1, 1, D, H, W)
    wm_prob_t = torch.from_numpy(wm_prob.astype(np.float32)).to(device).reshape(1, 1, D, H, W)

    # Precompute masks
    gm_mask = (seg_t == 2).float()  # [1, 1, D, H, W]
    wm_contour = extract_wm_contours(seg_t)  # [1, 1, D, H, W]
    # Active region: GM voxels + WM contour voxels (velocity is zero elsewhere)
    active_mask = (gm_mask + wm_contour).clamp(max=1.0)  # [1, 1, D, H, W]
    identity_grid = _make_identity_grid(shape, device)

    if save_fields_dir:
        assert ref_img is not None, "ref_img required when save_fields_dir is set"
        os.makedirs(save_fields_dir, exist_ok=True)

    # Velocity field: accumulated deformation
    velocity_out = None
    velocity_field = torch.zeros(1, 3, D, H, W, device=device)
    # Integrated field persists across outer iterations (ANTs behavior)
    integrated_field = torch.zeros(1, 3, D, H, W, device=device)
    # Cortical thickness output
    cortical_thickness = torch.zeros(1, 1, D, H, W, device=device)

    n_gm_voxels = int(gm_mask.sum())
    start_time = time.time()

    for iteration in range(max_iterations):
        iter_start = time.time()

        # Reset per-iteration accumulators (but NOT integrated_field!)
        forward_incremental = torch.zeros(1, 3, D, H, W, device=device)
        inverse_field = torch.zeros(1, 3, D, H, W, device=device)
        hit_image = torch.zeros(1, 1, D, H, W, device=device)
        total_image = torch.zeros(1, 1, D, H, W, device=device)
        thickness_image = torch.zeros(1, 1, D, H, W, device=device)

        # Collectors for field snapshots (only when saving). The per-integration
        # -point GPU->CPU copy below is the dominant cost of writing fields, so
        # only collect on an iteration whose snapshots are actually consumed:
        # save_fields_dir writes every iteration, velocity_field_prefix only the
        # last. Collecting on all 45 and using one was ~3x the per-iteration
        # cost for nothing.
        need_snapshots = bool(save_fields_dir) or (
            velocity_field_prefix is not None and cumulative_fields
            and iteration == max_iterations - 1)
        if need_snapshots:
            inverse_field_snapshots = []
            forward_field_snapshots = []

        # ---- Inner integration loop ----
        for pt in range(1, num_integration_points + 1):

            # Compose inverse field: inverse = compose(velocity, inverse)
            inverse_field = compose_fields(
                velocity_field * active_mask, inverse_field, identity_grid)

            # Warp images by inverse field
            warped_wm = warp_image(wm_prob_t, inverse_field, identity_grid)
            warped_wm_contours = warp_image(wm_contour, inverse_field, identity_grid)
            warped_thickness = warp_image(thickness_image, inverse_field, identity_grid)

            # Gradient of warped WM probability (smoothed, in voxel space)
            grad = gaussian_gradient_3d(warped_wm, grad_sigma_vox, device)

            # Normalize gradient direction in PHYSICAL space, then convert
            # back to voxel-space direction for the displacement field.
            if voxel_size_t is not None:
                # grad is d/d(voxel). Physical gradient = grad / voxel_size
                grad_phys = grad / voxel_size_t
                grad_phys_mag = (grad_phys * grad_phys).sum(dim=1, keepdim=True).sqrt()
                # Unit direction in physical space
                phys_dir = grad_phys / (grad_phys_mag + 1e-8)
                phys_dir = phys_dir * (grad_phys_mag > gradient_gate).float()
                # Convert physical direction to voxel displacement:
                # 1mm in direction d_i → 1/voxel_size_i voxels
                grad_safe = phys_dir / voxel_size_t
            else:
                grad_mag = (grad * grad).sum(dim=1, keepdim=True).sqrt()
                grad_safe = grad / (grad_mag + 1e-8)
                grad_safe = grad_safe * (grad_mag > gradient_gate).float()

            # Speed: -(warped_wm - gm_prob) * gm_prob * gradient_step at GM voxels
            # gradient_step is in mm; grad_safe is in voxels/mm, so the product
            # (grad_safe * speed) is in voxels — correct for the displacement field.
            delta = warped_wm - gm_prob_t
            speed = -delta * gm_prob_t * gradient_step  # [B, 1, D, H, W]
            speed = speed * gm_mask  # only at GM voxels

            # NaN protection
            speed = torch.where(torch.isfinite(speed), speed, torch.zeros_like(speed))

            # Update forward incremental field (voxel-space displacement)
            forward_incremental = forward_incremental + grad_safe * speed

            # ---- Thickness accumulation ----
            if pt == 1:
                # At point 1: use integrated_field from END of previous outer
                # iteration (carries full N-step displacement). ANTs sets
                # thicknessImage BEFORE resetting integratedField.
                if voxel_size_t is not None:
                    # Weighted magnitude: sqrt(sum((disp_i * voxsize_i)^2))
                    disp_mm = integrated_field * voxel_size_t
                    disp_mag = (disp_mm ** 2).sum(dim=1, keepdim=True).sqrt()
                else:
                    disp_mag = (integrated_field ** 2).sum(dim=1, keepdim=True).sqrt()
                thickness_image = disp_mag * wm_contour

                # Initialize accumulators (for GM and WM voxels)
                hit_image = wm_contour.clone()
                total_image = thickness_image.clone()
            else:
                # At later points: accumulate warped contours/thickness at GM
                hit_image = hit_image + warped_wm_contours * gm_mask
                total_image = total_image + warped_thickness * gm_mask

            # Constrain fields to active region
            inverse_field = inverse_field * active_mask
            velocity_field = velocity_field * active_mask

            # Reset integrated field at point 1 (AFTER thickness was set)
            if pt == 1:
                integrated_field.zero_()

            # Mutual field inversion for forward/inverse consistency
            integrated_field = invert_field(
                inverse_field, identity_grid,
                max_iter=max_invert_iterations, initial=integrated_field)
            inverse_field = invert_field(
                integrated_field, identity_grid,
                max_iter=max_invert_iterations, initial=inverse_field)

            if need_snapshots:
                # [3, D, H, W] snapshots — GPU→CPU transfer per integration point
                inverse_field_snapshots.append(inverse_field[0].cpu().numpy())
                forward_field_snapshots.append(integrated_field[0].cpu().numpy())

        # ---- Save fields for this iteration ----
        if save_fields_dir:
            affine = ref_img.affine
            # Stack integration points into 5D: [3, D, H, W, T]
            inv_4d = np.stack(inverse_field_snapshots, axis=-1)  # [3, D, H, W, T]
            fwd_4d = np.stack(forward_field_snapshots, axis=-1)  # [3, D, H, W, T]
            for name, data in [('inverse_field', inv_4d), ('forward_field', fwd_4d)]:
                # Save each component (d, h, w) as separate 4D volume [D, H, W, T]
                for comp_idx, comp_name in enumerate(['d', 'h', 'w']):
                    fname = os.path.join(save_fields_dir,
                                         f'{name}_{comp_name}_iter{iteration:03d}.nii.gz')
                    img = nib.Nifti1Image(data[comp_idx], affine)
                    img.header['xyzt_units'] = 2  # mm
                    nib.save(img, fname)
            if verbose:
                print(f"    Saved fields for iteration {iteration + 1}")

        if velocity_field_prefix and iteration == max_iterations - 1:
            if cumulative_fields:
                _save_velocity_fields(velocity_field_prefix, ref_img,
                                      inverse_field_snapshots, forward_field_snapshots)
            # the underlying velocity field, which the integration composes
            # num_integration_points times; [D, H, W, 3] in voxels (d, h, w)
            vel = velocity_field[0].cpu().numpy().transpose(1, 2, 3, 0).astype(np.float32)
            velocity_out = vel
            vimg = nib.Nifti1Image(vel, ref_img.affine)
            vimg.header['xyzt_units'] = 10
            nib.save(vimg, '{}Velocity.nii.gz'.format(velocity_field_prefix))

        # ---- After inner loop: update velocity and thickness ----

        # Smooth hit and total images
        smooth_hit = gaussian_smooth_3d(
            hit_image, smooth_sigma_vox, device, zero_boundary=False)
        smooth_total = gaussian_smooth_3d(
            total_image, smooth_sigma_vox, device, zero_boundary=False)

        # Update velocity field
        velocity_field = velocity_field + forward_incremental

        # Compute thickness and apply thickness prior constraint
        has_hits = (smooth_hit > 0.001)
        thickness_vals = torch.where(
            has_hits,
            smooth_total / smooth_hit.clamp(min=0.001),
            torch.zeros_like(smooth_hit))
        thickness_vals = thickness_vals.clamp(min=0)

        # Where thickness > prior at GM voxels, scale down velocity by (prior/thickness)^2
        tp = thickness_prior  # thickness_vals is already in mm when voxel_size is set
        over_prior = has_hits & (thickness_vals > tp) & (gm_mask > 0)
        if over_prior.any():
            fraction = tp / thickness_vals.clamp(min=1e-8)
            scale = torch.where(over_prior, fraction * fraction,
                                torch.ones_like(fraction))
            velocity_field = velocity_field * scale

        # Set cortical thickness at GM voxels
        cortical_thickness = thickness_vals * gm_mask

        # Smooth velocity field (ANTs uses replicate-pad, no boundary zeroing)
        sigma_now = vel_sigma_vox
        if velocity_smooth_sigma_start:
            frac = iteration / max(max_iterations - 1, 1)
            # power > 1 holds the sigma near the start value for longer
            frac = frac ** velocity_smooth_sigma_power
            sigma_now = velocity_smooth_sigma_start * (
                (vel_sigma_vox / velocity_smooth_sigma_start) ** frac)
        if velocity_smooth_selective is not None:
            velocity_field = selective_masked_smooth_3d(
                velocity_field, sigma_now, device, mode=velocity_smooth_mask_mode,
                coherence_threshold=velocity_smooth_selective,
                gate=velocity_smooth_gate)
        elif velocity_smooth_masked:
            velocity_field = direction_masked_smooth_3d(
                velocity_field, sigma_now, device, mode=velocity_smooth_mask_mode)
        else:
            velocity_field = gaussian_smooth_3d(
                velocity_field, sigma_now, device, zero_boundary=False)

        # Constrain to active region
        velocity_field = velocity_field * active_mask

        iter_time = time.time() - iter_start
        if verbose:
            mean_thick = float(cortical_thickness[gm_mask > 0].mean()) if n_gm_voxels > 0 else 0
            print(f"  Iteration {iteration + 1}/{max_iterations}: "
                  f"mean_thickness={mean_thick:.3f}mm, "
                  f"time={iter_time:.1f}s")

    total_time = time.time() - start_time
    if verbose:
        print(f"DiReCT CUDA: completed in {total_time:.1f}s")

    thickness_np = cortical_thickness.squeeze().cpu().numpy()
    # `velocity_out` is the same array just written to Velocity.nii.gz. Handing
    # it back lets callers skip re-reading (and gunzipping) the file they asked
    # for a moment earlier -- see solve_velocity_field.
    return (thickness_np, velocity_out) if return_velocity else thickness_np


def _try_compile():
    """Apply torch.compile to hot functions if available (PyTorch 2.0+).

    Compiled kernels are cached to ~/.cache/dldirect/torch_inductor/ so that
    subsequent runs skip the compilation overhead (~7s → <1s startup).
    """
    global gaussian_gradient_3d, gaussian_smooth_3d, warp_image
    global compose_fields, invert_field
    try:
        # Use a persistent cache directory (default /tmp is wiped on reboot)
        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache",
                                 "dldirect", "torch_inductor")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", cache_dir)

        gaussian_gradient_3d = torch.compile(gaussian_gradient_3d, dynamic=True)
        gaussian_smooth_3d = torch.compile(gaussian_smooth_3d, dynamic=True)
        warp_image = torch.compile(warp_image, dynamic=True)
        compose_fields = torch.compile(compose_fields, dynamic=True)
        # invert_field is not compiled: its early-stopping break creates
        # a new graph for each exit iteration, causing excessive recompilation.
        # The inner warp_image calls are still compiled individually.
    except Exception:
        pass  # older PyTorch or unsupported backend


_try_compile()
