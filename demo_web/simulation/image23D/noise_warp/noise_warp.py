"""Noise warping utilities — stripped to runtime-only components.

Only NoiseWarper and mix_new_noise (plus their helpers) are included.
Preprocessing utilities (get_noise_from_video, RAFT, background remover)
have been removed.
"""

import numpy as np
import torch
from einops import rearrange
import rp


# ---------------------------------------------------------------------------
# Low-level pixel helpers
# ---------------------------------------------------------------------------

def unique_pixels(image):
    c, h, w = image.shape
    pixels = rearrange(image, "c h w -> h w c")
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")
    unique_colors, inverse_indices, counts = torch.unique(
        flattened_pixels, dim=0, return_inverse=True, return_counts=True, sorted=False
    )
    u = unique_colors.shape[0]
    index_matrix = rearrange(inverse_indices, "(h w) -> h w", h=h, w=w)
    assert unique_colors.shape == (u, c)
    assert counts.shape == (u,)
    assert index_matrix.shape == (h, w)
    return unique_colors, counts, index_matrix


def sum_indexed_values(image, index_matrix):
    c, h, w = image.shape
    u = index_matrix.max() + 1
    pixels = rearrange(image, "c h w -> h w c")
    flattened_pixels = rearrange(pixels, "h w c -> (h w) c")
    output = torch.zeros((u, c), dtype=flattened_pixels.dtype, device=flattened_pixels.device)
    output.index_add_(0, index_matrix.view(-1), flattened_pixels)
    return output


def indexed_to_image(index_matrix, unique_colors):
    h, w = index_matrix.shape
    u, c = unique_colors.shape
    flattened_image = unique_colors[index_matrix.view(-1)]
    image = rearrange(flattened_image, "(h w) c -> h w c", h=h, w=w)
    image = rearrange(image, "h w c -> c h w")
    return image


_arange_cache = {}


def _cached_arange(length, device, dtype):
    code = hash((length, device, dtype))
    if code in _arange_cache:
        return _arange_cache[code]
    _arange_cache[code] = torch.arange(length, device=device, dtype=dtype)
    return _arange_cache[code]


def fast_nearest_torch_remap_image(image, x, y, *, relative=False, add_alpha_mask=False):
    in_c, in_height, in_width = image.shape
    out_height, out_width = x.shape

    if add_alpha_mask:
        alpha_mask = torch.ones_like(image[:1])
        image = torch.cat([image, alpha_mask], dim=0)

    if torch.is_floating_point(x):
        x = x.round_().long()
    if torch.is_floating_point(y):
        y = y.round_().long()

    if relative:
        x += _cached_arange(in_width, device=x.device, dtype=x.dtype)
        y += _cached_arange(in_height, device=y.device, dtype=y.dtype)[:, None]

    x.clamp_(0, in_width - 1)
    y.clamp_(0, in_height - 1)
    out = image[:, y, x]
    return out


def regaussianize(noise):
    c, hs, ws = noise.shape
    unique_colors, counts, index_matrix = unique_pixels(noise[:1])
    u = len(unique_colors)
    foreign_noise = torch.randn_like(noise)
    summed_foreign_noise_colors = sum_indexed_values(foreign_noise, index_matrix)
    meaned_foreign_noise_colors = summed_foreign_noise_colors / rearrange(counts, "u -> u 1")
    meaned_foreign_noise = indexed_to_image(index_matrix, meaned_foreign_noise_colors)
    zeroed_foreign_noise = foreign_noise - meaned_foreign_noise
    counts_as_colors = rearrange(counts, "u -> u 1")
    counts_image = indexed_to_image(index_matrix, counts_as_colors)
    output = noise
    output = output / counts_image ** .5
    output = output + zeroed_foreign_noise
    return output, counts_image


# ---------------------------------------------------------------------------
# State / flow helpers
# ---------------------------------------------------------------------------

@rp.memoized
def _xy_meshgrid(h, w, device, dtype):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))
    output = torch.stack([x, y]).to(device, dtype)
    assert output.shape == (2, h, w)
    return output


def xy_meshgrid_like_image(image):
    assert image.ndim == 3, "image is in CHW form"
    c, h, w = image.shape
    return _xy_meshgrid(h, w, image.device, image.dtype)


def noise_to_xyωc(noise):
    assert noise.ndim == 3, "noise is in CHW form"
    zeros = torch.zeros_like(noise[0][None])
    ones = torch.ones_like(noise[0][None])
    return torch.concat([zeros, zeros, ones, noise])


def xyωc_to_noise(xyωc):
    assert xyωc.ndim == 3
    assert xyωc.shape[0] > 3
    return xyωc[3:]


def warp_xyωc(I, F, xy_mode="none", expand_only=False, index=None):
    assert F.device == I.device
    assert F.ndim == 3
    assert I.ndim == 3
    xyωc, h, w = I.shape
    assert F.shape == (2, h, w)
    device = I.device

    x = 0
    y = 1
    xy = 2
    xyω = 3
    ω = 2
    c = xyωc - xyω
    ωc = xyωc - xy
    w_dim = 2
    assert c, 'I has no noise channels.'
    assert (I[ω] > 0).all()

    grid = xy_meshgrid_like_image(I)

    init = torch.empty_like(I)
    init[:xy] = 0
    init[ω] = 1
    init[-c:] = 0

    pre_expand = torch.empty_like(I)

    interp = 'nearest' if not isinstance(expand_only, str) else expand_only
    regauss = not isinstance(expand_only, str)

    pre_expand[:xy] = rp.torch_remap_image(I[:xy], *-F, relative=True, interp=interp)
    pre_expand[-ωc:] = rp.torch_remap_image(I[-ωc:], *-F, relative=True, interp=interp)
    pre_expand[ω][pre_expand[ω] == 0] = 1

    if expand_only:
        if regauss:
            pre_expand[-c:] = regaussianize(pre_expand[-c:])[0]
        else:
            pre_expand[-c:] = torch.randn_like(pre_expand[-c:]) * (pre_expand[-c:] == 0) + pre_expand[-c:]
        return pre_expand

    pre_shrink = I.clone()
    pre_shrink[:xy] += F

    pos = (grid + pre_shrink[:xy]).round()
    in_bounds = (0 <= pos[x]) & (pos[x] < w) & (0 <= pos[y]) & (pos[y] < h)
    in_bounds = in_bounds[None]
    out_of_bounds = ~in_bounds
    pre_shrink = torch.where(out_of_bounds, init, pre_shrink)

    scat_xy = pre_shrink[:xy].round()
    pre_shrink[:xy] -= scat_xy

    assert xy_mode in ['float', 'none'] or isinstance(xy_mode, int)
    if xy_mode == 'none':
        pre_shrink[:xy] = 0

    if isinstance(xy_mode, int):
        quant = xy_mode
        pre_shrink[:xy] = (pre_shrink[:xy] * quant).round() / quant

    scat = lambda tensor: rp.torch_scatter_add_image(tensor, *scat_xy, relative=True)

    shrink_mask = torch.ones(1, h, w, dtype=bool, device=device)
    shrink_mask = scat(shrink_mask)
    assert shrink_mask.dtype == torch.bool

    pre_expand = torch.where(shrink_mask, init, pre_expand)

    concat_dim = w_dim
    concat = torch.concat([pre_shrink, pre_expand], dim=concat_dim)

    concat[-c:], counts_image = regaussianize(concat[-c:])

    concat[ω] /= counts_image[0]
    concat[ω] = concat[ω].nan_to_num()

    pre_shrink, expand = torch.chunk(concat, chunks=2, dim=concat_dim)

    shrink = torch.empty_like(pre_shrink)
    shrink[ω] = scat(pre_shrink[ω][None])[0]
    shrink[:xy] = scat(pre_shrink[:xy] * pre_shrink[ω][None]) / shrink[ω][None]
    shrink[-c:] = scat(pre_shrink[-c:] * pre_shrink[ω][None]) / scat(pre_shrink[ω][None] ** 2).sqrt()

    output = torch.where(shrink_mask, shrink, expand)
    output[ω] = output[ω] / output[ω].mean()
    ε = .00001
    output[ω] += ε
    assert (output[ω] > 0).all()
    output[ω] **= .9999

    return output


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def blend_noise(noise_background, noise_foreground, alpha):
    """Variance-preserving blend."""
    return (noise_foreground * alpha + noise_background * (1 - alpha)) / (alpha ** 2 + (1 - alpha) ** 2) ** .5


def mix_new_noise(noise, alpha):
    """As alpha --> 1, noise is destroyed."""
    if isinstance(noise, torch.Tensor):
        return blend_noise(noise, torch.randn_like(noise), alpha)
    elif isinstance(noise, np.ndarray):
        return blend_noise(noise, np.random.randn(*noise.shape), alpha)
    else:
        raise TypeError(f"Unsupported input type: {type(noise)}.")


class NoiseWarper:
    def __init__(
        self, c, h, w, device,
        dtype=torch.float32,
        scale_factor=1,
        post_noise_alpha=0,
        progressive_noise_alpha=0,
        warp_kwargs=dict(),
    ):
        self.c = c
        self.h = h
        self.w = w
        self.device = device
        self.dtype = dtype
        self.scale_factor = scale_factor
        self.progressive_noise_alpha = progressive_noise_alpha
        self.post_noise_alpha = post_noise_alpha
        self.warp_kwargs = warp_kwargs

        self._state = self._noise_to_state(
            noise=torch.randn(
                c,
                h * scale_factor,
                w * scale_factor,
                dtype=dtype,
                device=device,
            )
        )

    @property
    def noise(self):
        noise = self._state_to_noise(self._state)
        weights = self._state[2][None]
        noise = (
            rp.torch_resize_image(noise * weights, (self.h, self.w), interp="area")
            / rp.torch_resize_image(weights ** 2, (self.h, self.w), interp="area").sqrt()
        )
        noise = noise * self.scale_factor

        if self.post_noise_alpha:
            noise = mix_new_noise(noise, self.post_noise_alpha)

        return noise

    def __call__(self, dx, dy, idx=None):
        if rp.is_numpy_array(dx):
            dx = torch.tensor(dx).to(self.device, self.dtype)
        if rp.is_numpy_array(dy):
            dy = torch.tensor(dy).to(self.device, self.dtype)

        flow = torch.stack([dx, dy]).to(self.device, self.dtype)
        _, oflowh, ofloww = flow.shape

        assert flow.ndim == 3 and flow.shape[0] == 2
        flow = rp.torch_resize_image(
            flow,
            (self.h * self.scale_factor, self.w * self.scale_factor),
        )

        _, flowh, floww = flow.shape
        flow[0] *= flowh / oflowh * self.scale_factor
        flow[1] *= floww / ofloww * self.scale_factor

        self._state = self._warp_state(self._state, flow, index=idx)
        return self

    @staticmethod
    def _noise_to_state(noise):
        return noise_to_xyωc(noise)

    @staticmethod
    def _state_to_noise(state):
        return xyωc_to_noise(state)

    def _warp_state(self, state, flow, index=None):
        if self.progressive_noise_alpha:
            state[3:] = mix_new_noise(state[3:], self.progressive_noise_alpha)
        return warp_xyωc(state, flow, **self.warp_kwargs, index=index)
