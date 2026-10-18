"""Upload validation and privacy-safe file metadata (Pillow only; no model needed).

What the detectors actually see (official OmniAID data/datasets.py at commit 40749406, unchanged here):
  - only files ending in .png/.jpg/.jpeg/.webp (case-insensitive) are read;
  - Image.open(path).convert('RGB'): alpha is dropped, greyscale/palette/CMYK are converted, animated files give frame 1;
  - NO EXIF orientation is applied: the model sees the pixels in their STORED orientation;
  - Resize to 336x336 (aspect ratio not kept), ToTensor, CLIP mean/std normalisation;
  - the module sets ImageFile.LOAD_TRUNCATED_IMAGES = True, and an unreadable file is silently replaced by a random
    other file in the batch. The platform therefore checks every upload BEFORE it reaches the model and rejects
    unreadable or truncated files, so neither behaviour can affect a result.
Validation never modifies the uploaded file; the model reads the original bytes.

Privacy: GPS values are never read into the results. Only "GPS metadata present: Yes/No" is reported.
Owner, artist, copyright, serial-number and maker-note fields are not extracted.
"""
from pathlib import Path

from PIL import ExifTags, Image, ImageFile

from config import MAX_FILE_MB, MODEL_EXTENSIONS

METADATA_NOTE = ("File metadata is contextual information only. It can be modified, removed, copied or fabricated, "
                 "and many platforms strip it on upload, so it is not evidence that an image is authentic or AI-generated.")

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825


def _fmt_ratio(v):
    try:
        return float(v)
    except Exception:
        return None


def _exposure(v):
    f = _fmt_ratio(v)
    if not f:
        return ""
    return f"1/{round(1 / f)} s" if f < 1 else f"{f:g} s"


def validate_image(path):
    """Return (ok, processing_status, note). Never alters the file."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in MODEL_EXTENSIONS:
        return False, "UNSUPPORTED", f"File type '{ext or 'none'}' is not supported. Supported: JPEG, PNG, WebP."
    size = p.stat().st_size
    if size == 0:
        return False, "FAILED", "Empty file."
    if size > MAX_FILE_MB * 1024 * 1024:
        return False, "UNSUPPORTED", f"File larger than {MAX_FILE_MB:g} MB."
    old = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # the model module turns this on globally; validation must be strict
    try:
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()  # full decode: catches truncated or corrupt pixel data
            if im.width < 8 or im.height < 8:
                return False, "UNSUPPORTED", f"Image too small ({im.width}x{im.height})."
            note = "Animated/multi-frame file: the detectors analyse the first frame only." if getattr(im, "n_frames", 1) > 1 else ""
    except Exception as e:
        return False, "FAILED", f"Unreadable or corrupt image ({type(e).__name__})."
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old
    return True, "ANALYSED", note


def extract_metadata(path, display_name=None):
    """Safe technical metadata. Never raises; unreadable files return the basic file fields only."""
    p = Path(path)
    md = {"filename": display_name or p.name, "format": "", "dimensions": "", "width": None, "height": None,
          "file_size_bytes": p.stat().st_size if p.exists() else None, "colour_mode": "", "exif_present": False,
          "camera_make": "", "camera_model": "", "capture_datetime": "", "iso": "", "exposure_time": "",
          "f_number": "", "focal_length": "", "software": "", "orientation_tag": "", "gps_metadata_present": False,
          "other_embedded_metadata": ""}
    try:
        with Image.open(p) as im:
            md.update(format=im.format or "", width=im.width, height=im.height, dimensions=f"{im.width}x{im.height}",
                      colour_mode=im.mode)
            exif = im.getexif()
            if exif and len(exif):
                md["exif_present"] = True
                sub = exif.get_ifd(_EXIF_IFD)
                get = lambda name: sub.get(_TAGS[name], exif.get(_TAGS[name]))  # noqa: E731
                md["camera_make"] = str(get("Make") or "").strip("\x00 ")
                md["camera_model"] = str(get("Model") or "").strip("\x00 ")
                md["capture_datetime"] = str(get("DateTimeOriginal") or get("DateTime") or "").strip("\x00 ")
                iso = get("ISOSpeedRatings")
                md["iso"] = str(iso[0] if isinstance(iso, tuple) else iso or "")
                md["exposure_time"] = _exposure(get("ExposureTime"))
                fn = _fmt_ratio(get("FNumber"))
                md["f_number"] = f"f/{fn:g}" if fn else ""
                fl = _fmt_ratio(get("FocalLength"))
                md["focal_length"] = f"{fl:g} mm" if fl else ""
                md["software"] = str(get("Software") or "").strip("\x00 ")[:60]
                o = exif.get(_TAGS["Orientation"])
                md["orientation_tag"] = str(o) if o else ""
                md["gps_metadata_present"] = bool(exif.get_ifd(_GPS_IFD)) or _GPS_IFD in exif
            other = [k for k in ("xmp", "XML:com.adobe.xmp", "icc_profile", "parameters", "prompt", "workflow", "Comment", "comment")
                     if k in im.info]
            other += [f"PNG text: {k}" for k in getattr(im, "text", {}) if k not in other][:5] if im.format == "PNG" else []
            md["other_embedded_metadata"] = ", ".join(other)
    except Exception:
        pass
    return md


def orientation_note(md):
    if md.get("orientation_tag") and md["orientation_tag"] != "1":
        return (f"This file carries an EXIF orientation tag ({md['orientation_tag']}). The detectors analyse the pixels in their "
                "stored orientation, which is what the preview shows; other image viewers may display it rotated.")
    return ""


def make_preview(path, out_dir, max_side=768):
    """Display-only copy in the STORED pixel orientation (same orientation the model sees). Never used for scoring."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(path).stem[:80] + "_preview.jpg")
    i = 1
    while out.exists():
        out = out_dir / (Path(path).stem[:80] + f"_preview_{i}.jpg"); i += 1
    with Image.open(path) as im:
        im = im.convert("RGB")  # no ImageOps.exif_transpose on purpose
        im.thumbnail((max_side, max_side))
        im.save(out, "JPEG", quality=90)  # saved without EXIF, so browsers cannot re-rotate it
    return str(out)


def human_size(n):
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
