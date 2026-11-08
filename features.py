"""Feature mining for AI-generated image detection.

Two feature families (prefixes keep them separable for ablation studies):
  meta_*  -> forensic cues mined from EXIF / PNG text chunks / embedded provenance tags
  vis_*   -> handcrafted statistical + structural cues from pixels
"""
from __future__ import annotations

import io #read image data directly from memory ising BytesIO
import os
from typing import Union #function can accept different data types

import cv2
"""
Used for image processing operations such as 
resizing, grayscale conversion, edge detection, noise analysis
"""
import numpy as np
from PIL import ExifTags, Image, ImageOps
"""
Python Imaging Library- open imagess and extract metadata 
such as EXIF

EXIF tag numbers converted to meaningful names 
such as camera make and model

ImageOps for correcting image orientation

"""
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
#calculate glcm texture features
#local_binary_pattern for extracting lbp textual features

Source = Union[str, os.PathLike, bytes]
"""
File path as a string
File path using os.Pathlike
Image directly as bytes -> given as input
"""
NORM = 1024  # every image is centre-squared and resized to this before visual features are measured
CROP = 512  # centre crop (no resizing -> noise statistics stay untouched) -> for visual feature
BLOCK = 32 # for noise and tecture uniformity detection

AI_TOOL_KEYWORDS = (
    "stable diffusion", "stablediffusion", "midjourney", "dall-e", "dalle", "firefly",
    "comfyui", "automatic1111", "novelai", "leonardo", "dreamstudio", "imagen",
    "flux", "sdxl", "invokeai", "playground", "ideogram", "generative", "diffusion",
    "openai", "gemini", "grok",
)
AI_TEXT_CHUNK_KEYS = ("parameters", "prompt", "workflow", "negative prompt", "sd-metadata", "dream")
#metadata fields with possible embeddings
COMMON_AI_SIZES = {256, 512, 640, 768, 832, 896, 1024, 1152, 1216, 1344, 1536, 2048}
# image ddimensions commonly used by ai
_TAGS = ExifTags.TAGS

#reads image and converts it to bytes
def _read_bytes(src: Source) -> bytes:
    if isinstance(src, (bytes, bytearray)): #if already in bytes
        return bytes(src)
    with open(src, "rb") as fh: #rb read binary 
        return fh.read() #read entire file

#extract EXIF metadata
def _exif_dict(img: Image.Image) -> dict:
    out: dict = {}
    try:
        exif = img.getexif()
    except Exception:
        return out
    for k, v in exif.items(): #exif data stored using numerical tag IDs
        out[_TAGS.get(k, k)] = v #converst to readable names called tags
    for ifd_id in (0x8769,):  # Exif sub-IFD (lens, exposure, original datetime ...)- Image file directory
        try:
            for k, v in exif.get_ifd(ifd_id).items():
                out[_TAGS.get(k, k)] = v #adds to dictionary
        except Exception:
            pass
    try:
        gps = exif.get_ifd(0x8825)
        if gps:
            out["GPSInfo"] = dict(gps) #adds to dictionary as GPSInfo
    except Exception:
        pass
    return out #returns complete EXIF dictionary

#cleaning metadata values
def _clean(v, n=120) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "ignore")  #cinverts byte into readable text
    return str(v).strip().replace("\x00", "")[:n]


# --------------------------------------------------------------------------- metadata
def extract_metadata(data: bytes) -> tuple[dict, dict]:
    """Returns (numeric feature dict, human-readable summary dict)."""
    with Image.open(io.BytesIO(data)) as img: #open image 
        w, h = img.size
        fmt = (img.format or "").upper()
        exif = _exif_dict(img)
        info = {str(k): v for k, v in img.info.items()}
        has_icc = int("icc_profile" in info) #if icc colour profile

    make, model = _clean(exif.get("Make", "")), _clean(exif.get("Model", ""))
    lens = _clean(exif.get("LensModel", exif.get("LensMake", "")))
    dt = _clean(exif.get("DateTimeOriginal", exif.get("DateTime", "")))
    software = _clean(exif.get("Software", info.get("Software", "")))
    has_gps = int(bool(exif.get("GPSInfo")))
    """
    1- GPS INFO EXISTS
    0 - GPS INFO DOESNT EXIST
    """
    text_blob = " ".join(
        _clean(v, 2000) for k, v in info.items()
        if isinstance(v, (str, bytes)) and k not in ("icc_profile", "exif", "xmp")
    ).lower() + " " + software.lower()
    """
    Iterated through fields, 
    Select values in string or bytes, 
    Excludes icc, exif, xmp fields
    Cleans
    Joins them to one string- lower case
    Adds software information  -> to search for ai generation keywords
    """
    has_ai_text_chunk = int(any(k.lower() in AI_TEXT_CHUNK_KEYS for k in info))
    ai_keyword = int(any(kw in text_blob for kw in AI_TOOL_KEYWORDS))

    head = data[:262144].lower() + data[-65536:].lower()
    has_c2pa = int(b"c2pa" in head or b"trainedalgorithmicmedia" in head or b"compositewithtrainedalgorithmicmedia" in head)
    if not ai_keyword and b"trainedalgorithmicmedia" in head:
        ai_keyword = 1
    """
    Contains numerical values for the model
    """
    feats = {
        "meta_has_exif": int(len(exif) > 0), #whether exif data is there
        "meta_exif_count": len(exif), # number of fields
        "meta_has_make": int(bool(make)), #camera manufacturer
        "meta_has_model": int(bool(model)), #model
        "meta_has_lens": int(bool(lens)), #lens information
        "meta_has_datetime": int(bool(dt)), #date 
        "meta_has_gps": has_gps, #gps
        "meta_has_software": int(bool(software)), #software
        "meta_software_ai_keyword": ai_keyword,
        "meta_has_ai_text_chunk": has_ai_text_chunk, #prompt/parameter fields
        "meta_has_c2pa_or_iptc_ai": has_c2pa,
        "meta_has_icc": has_icc,
        "meta_is_jpeg": int(fmt in ("JPEG", "MPO")),
        "meta_is_png": int(fmt == "PNG"),
        "meta_width": w,
        "meta_height": h,
        "meta_megapixels": w * h / 1e6,
        "meta_aspect_ratio": w / max(h, 1),
        "meta_dims_ai_typical": int(w in COMMON_AI_SIZES and h in COMMON_AI_SIZES),
        "meta_file_kb": len(data) / 1024,
        "meta_bytes_per_pixel": len(data) / max(w * h, 1),
    } 
    """
    Human readable informations for user interface
    """
    summary = {
        "Format": fmt, "Dimensions": f"{w} x {h}", "File size": f"{len(data)/1024:.0f} KB",
        "Camera make": make or "-", "Camera model": model or "-", "Lens": lens or "-",
        "Date/time": dt or "-", "Software": software or "-",
        "GPS present": "yes" if has_gps else "no",
        "EXIF fields": len(exif),
        "AI-tool keyword in metadata": "yes" if ai_keyword else "no",
        "Generation text chunk (prompt/parameters)": "yes" if has_ai_text_chunk else "no",
        "C2PA / IPTC AI-source tag": "yes" if has_c2pa else "no",
    }
    return feats, summary


# --------------------------------------------------------------------------- visual
def _entropy(arr: np.ndarray, bins: int = 256, rng=(0, 256)) -> float:
    hist, _ = np.histogram(arr, bins=bins, range=rng) #pixel internsity range into 256 bins
    p = hist / max(hist.sum(), 1) #convert to probabilities
    p = p[p > 0] #remove zero probabilties
    return float(-(p * np.log2(p)).sum()) #calulcate entropy- numerical measure of pixel intensity variation 

"""
Divides image into 32x32 blocks
each block is examines for noise and texture
"""
def _block_view(a: np.ndarray, b: int = BLOCK):
    bh, bw = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    if bh == 0 or bw == 0:
        return None
    return a[:bh, :bw].reshape(bh // b, b, bw // b, b)
"""
All images consistent in size
if 1920x1080 then take 1080x1080 square from centre
"""

def normalize_scale(rgb: np.ndarray) -> np.ndarray:
    """Centre-square then resize to NORM x NORM so noise/texture statistics are comparable across sources."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = np.ascontiguousarray(rgb[y0:y0 + s, x0:x0 + s])
    if s == NORM:
        return sq
    return cv2.resize(sq, (NORM, NORM), interpolation=cv2.INTER_AREA if s > NORM else cv2.INTER_CUBIC)
    #interpolation methods for resizing

def extract_visual(rgb: np.ndarray) -> dict:
    rgb = normalize_scale(rgb)
    h, w = rgb.shape[:2]
    ch, cw = min(h, CROP), min(w, CROP)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    rgb = np.ascontiguousarray(rgb[y0:y0 + ch, x0:x0 + cw])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32) #grayscale pixel values to floating point numbers
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) #rgp -> hsv
    """
    HSV represents colour using hue, saturation, value for brightness
    """
    f: dict = {}

    # global tone
    f["vis_brightness"] = float(g.mean())
    f["vis_contrast"] = float(g.std())
    f["vis_entropy"] = _entropy(gray)

    # sharpness & edges
    f["vis_sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var()) #laplacian filerr
    f["vis_edge_density"] = float(cv2.Canny(gray, 100, 200).mean() / 255.0)

    # sensor-noise proxy: residual after median denoise
    resid = g - cv2.medianBlur(gray, 3).astype(np.float32)
    f["vis_noise_std"] = float(resid.std())
    f["vis_noise_mad"] = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    rb = _block_view(resid)
    gb = _block_view(g)
    if rb is not None:
        bs = rb.std(axis=(1, 3)).ravel()
        f["vis_noise_uniformity"] = float(bs.std() / (bs.mean() + 1e-6))  # low => suspiciously even noise
        bc = gb.std(axis=(1, 3)).ravel()
        f["vis_texture_uniformity"] = float(bc.std() / (bc.mean() + 1e-6))
    else:
        f["vis_noise_uniformity"] = np.nan
        f["vis_texture_uniformity"] = np.nan

    # texture: GLCM + LBP
    q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 2], levels=64, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        f[f"vis_glcm_{prop}"] = float(graycoprops(glcm, prop).mean())
    lbp = local_binary_pattern(gray, 8, 1, "uniform")
    f["vis_lbp_entropy"] = _entropy(lbp, bins=10, rng=(0, 10))

    # colour
    f["vis_saturation_mean"] = float(hsv[..., 1].mean())
    f["vis_saturation_std"] = float(hsv[..., 1].std())
    for i, c in enumerate("rgb"):
        ch_ = rgb[..., i]
        f[f"vis_{c}_mean"] = float(ch_.mean())
        f[f"vis_{c}_std"] = float(ch_.std())
        f[f"vis_{c}_hist_entropy"] = _entropy(ch_)
        hist, _ = np.histogram(ch_, bins=8, range=(0, 256))
        hist = hist / hist.sum()
        for b in range(8):
            f[f"vis_hist_{c}{b}"] = float(hist[b])
    q5 = (rgb >> 3).reshape(-1, 3).astype(np.uint32)
    codes = (q5[:, 0] << 10) | (q5[:, 1] << 5) | q5[:, 2]
    f["vis_unique_color_ratio"] = float(len(np.unique(codes)) / codes.size)

    # frequency domain: share of spectral energy in mid/high bands
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    yy, xx = np.indices(spec.shape)
    r = np.hypot(yy - spec.shape[0] / 2, xx - spec.shape[1] / 2)
    R = min(spec.shape) / 2
    tot = spec.sum() + 1e-9
    f["vis_fft_mid_ratio"] = float(spec[(r > 0.25 * R) & (r <= 0.5 * R)].sum() / tot)
    f["vis_fft_high_ratio"] = float(spec[r > 0.5 * R].sum() / tot)
    return f


def extract_all(src: Source) -> tuple[dict, dict, np.ndarray] | None:
    """Returns (features, metadata summary, rgb array) or None if unreadable."""
    try:
        data = _read_bytes(src)
        meta, summary = extract_metadata(data)
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = np.asarray(img.convert("RGB"))
        if rgb.shape[0] < 16 or rgb.shape[1] < 16:
            return None
        vis = extract_visual(rgb)
        return {**meta, **vis}, summary, rgb
    except Exception:
        return None


def feature_row(src: Source) -> dict | None:
    """Light wrapper for batch extraction (no pixel array returned)."""
    out = extract_all(src)
    return None if out is None else out[0]

"""

Visual features include brightness, contrast, entropy, sharpness, noise statistics, glcm, lbp, colour histograms, fft based frequency featues.

1. Brightness- Avg pixel intensity -> Numpy Mean
2. Contrast - Pixel Intensity variation -> Standard Deviation
3. Entropy - Pixel Intensity Distrubution -> Shannon Entropy, Histogram
4. Sharpness -  Edge variance -> Laplacian Variance (openccv)
5. Edge Density - detect edges -> Canny Edge Detection (opencv)
6. Noise Detection - Original-Mean filtered -> Median Blur, Standard Deviation, MAD
7. Noise Uniformity -> Block-wise Statistical Analysis
8. GLCM Matrix - neighbouring pizel intensity relationship
9. LBP - compare neighbours, get local texture patterns -> Local Binary Pattern
10. Image Frequency Spectrum - 2D Fast Fourier Transform
"""