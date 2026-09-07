import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2, numpy as np
from filters import basic_filters as b
from filters import artistic_filters as a

rng = np.random.default_rng(1)
f = rng.integers(0, 256, (540, 960, 3), dtype=np.uint8)
big = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)

def bench(label, fn, frame, n=15):
    fn(frame); fn(frame)
    t = time.perf_counter()
    for _ in range(n): out = fn(frame)
    ms = (time.perf_counter() - t) * 1000 / n
    print(f"  {label:<38} {ms:7.2f} ms")
    return out

print("PIECES @ 960x540")
bench("cvtColor BGR2HSV", lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV), f)
bench("adjust_saturation (HSV round trip)", lambda x: b.adjust_saturation(x, 0.72), f)

def sat_luma(x, amount):
    gray = cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(x, amount, gray, 1.0 - amount, 0)
bench("saturation via luma blend", lambda x: sat_luma(x, 0.72), f)

bench("LUT 3ch", lambda x: cv2.LUT(x, b.WARM_LUT), f)
bench("convertScaleAbs", lambda x: cv2.convertScaleAbs(x, alpha=1.18, beta=-23), f)
bench("split/add/merge", lambda x: cv2.merge([cv2.subtract(c, 12) for c in cv2.split(x)]), f)
bench("vignette (float mask, current)", lambda x: a.apply_vignette(x, 0.45), f)

h, w = f.shape[:2]
mask = a._vignette_mask(h, w, 0.45)
mask_u8 = np.clip(mask * 255, 0, 255).astype(np.uint8)
mask_u8_3 = np.repeat(mask_u8, 3, axis=2)
bench("vignette (uint8 mask, cv2.multiply)", lambda x: cv2.multiply(x, mask_u8_3, scale=1/255.0), f)

mask16 = np.clip(mask * 4096, 0, 65535).astype(np.uint16)
mask16_3 = np.repeat(mask16, 3, axis=2)
bench("vignette (uint16 mask /4096)", lambda x: cv2.multiply(x, mask16_3, dtype=cv2.CV_8U, scale=1/4096.0), f)

bench("GaussianBlur sigma=10.7 full", lambda x: cv2.GaussianBlur(x, (0,0), sigmaX=x.shape[1]/90.0), f)
def blur_small(x):
    small = cv2.resize(x, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0,0), sigmaX=small.shape[1]/90.0)
    return cv2.resize(small, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_LINEAR)
bench("GaussianBlur on 1/4 then upscale", blur_small, f)

print("\nCURRENT FILTERS")
for label, fn in (("vintage", a.vintage), ("dreamy", a.dreamy), ("bright", b.bright)):
    bench(f"{label} @960", fn, f)
    bench(f"{label} @1920", fn, big)

# quality delta of the cheap vignette
ref = a.apply_vignette(f, 0.45).astype(np.int16)
alt = cv2.multiply(f, mask_u8_3, scale=1/255.0).astype(np.int16)
print(f"\nvignette uint8-vs-float max abs diff: {np.abs(ref-alt).max()}  mean: {np.abs(ref-alt).mean():.3f}")
alt16 = cv2.multiply(f, mask16_3, dtype=cv2.CV_8U, scale=1/4096.0).astype(np.int16)
print(f"vignette uint16-vs-float max abs diff: {np.abs(ref-alt16).max()}  mean: {np.abs(ref-alt16).mean():.3f}")
