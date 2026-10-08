"""YUV420p -> RGB conversion matching libswscale's x86 SIMD path (numpy emulation).

H.264 decoding is bit-exact, but OpenCV's FFmpeg backend converts YUV to BGR with
libswscale, whose fast (non-accurate_rnd) x86 MMX/SSSE3 path differs by about one
grey level from the C/ARM path used on Apple silicon. Official UCF VadCLIP
features were matched to fp16 backend noise only with this x86 emulation
(outputs/setup/ucf_yuv_conversion_20261008_134025, ucf_affine_search_*_x86_float16).
"""

import numpy as np


def _round16(value):
    return (value + (1 << 15)) >> 16


def _x86_coefficients():
    """ff_yuv2rgb_c_init_tables for BT.601 limited range, default contrast/saturation."""
    crv, cbu, cgu, cgv = 104597, 132201, -25675, -53279  # ff_yuv2rgb_coeffs[SWS_CS_ITU601]
    cy = (1 << 16) * 255 // 219
    oy = 16 << 16
    return {'y': _round16(cy * (1 << 13)), 'vr': _round16(crv * (1 << 13)), 'ub': _round16(cbu * (1 << 13)),
            'ug': _round16(cgu * (1 << 13)), 'vg': _round16(cgv * (1 << 13)), 'y_offset': _round16(oy * (1 << 3)),
            'uv_offset': 128 << 3}


X86_COEFF = _x86_coefficients()  # y=9539, vr=13075, ub=16525, ug=-3209, vg=-6660, y_offset=128


def pmulhw(a, b):
    return (a * b) >> 16


def yuv420p_to_rgb_x86(buffer, width, height):
    """16-bit pmulhw fixed point with nearest (replicated) chroma; returns RGB uint8 [H,W,3]."""
    if width % 2 or height % 2:
        raise ValueError('yuv420p emulation expects even frame dimensions')
    a = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
    luma = width * height
    if a.size != luma * 3 // 2:
        raise ValueError('Unexpected yuv420p buffer size')
    y = a[:luma].reshape(height, width)
    u = a[luma:luma * 5 // 4].reshape(height // 2, width // 2).repeat(2, 0).repeat(2, 1)
    v = a[luma * 5 // 4:].reshape(height // 2, width // 2).repeat(2, 0).repeat(2, 1)
    c = X86_COEFF
    u8, v8, y8 = (u << 3) - c['uv_offset'], (v << 3) - c['uv_offset'], (y << 3) - c['y_offset']
    yc = pmulhw(y8, c['y'])
    r = yc + pmulhw(v8, c['vr'])
    g = yc + pmulhw(u8, c['ug']) + pmulhw(v8, c['vg'])
    b = yc + pmulhw(u8, c['ub'])
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)
