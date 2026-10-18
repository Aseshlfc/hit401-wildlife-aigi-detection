"""File handling and metadata tests on GENERATED files only (no research images)."""
from PIL import Image

from metadata import extract_metadata, make_preview, orientation_note, validate_image

GPS_LAT = (12.0, 34.0, 56.78)


def _jpeg(p, size=(64, 48), exif=None):
    im = Image.new("RGB", size, (120, 140, 90))
    im.save(p, "JPEG", exif=exif) if exif is not None else im.save(p, "JPEG")
    return p


def _exif_with_gps(orientation=None):
    ex = Image.Exif()
    ex[0x010F] = "TestMake"; ex[0x0110] = "TestModel 1"; ex[0x0131] = "TestSoftware"
    ex[0x013B] = "Private Person"; ex[0x8298] = "Copyright Owner"       # Artist, Copyright: must NOT be extracted
    if orientation:
        ex[0x0112] = orientation
    sub = {}
    sub[0x9003] = "2024:05:06 07:08:09"; sub[0x8827] = 400; sub[0x829A] = 1 / 250; sub[0x829D] = 2.8; sub[0x920A] = 35.0
    sub[0xA431] = "SERIAL123"                                            # BodySerialNumber: must NOT be extracted
    ex[0x8769] = sub                                                     # Exif sub-IFD
    ex[0x8825] = {1: "S", 2: GPS_LAT, 3: "E", 4: (130.0, 50.0, 0.0)}   # GPS IFD
    return ex


def test_plain_jpeg_no_exif(tmp_path):
    p = _jpeg(tmp_path / "plain.jpg")
    assert validate_image(p) == (True, "ANALYSED", "")
    md = extract_metadata(p)
    assert md["exif_present"] is False and md["gps_metadata_present"] is False
    assert (md["format"], md["dimensions"], md["colour_mode"]) == ("JPEG", "64x48", "RGB")


def test_exif_fields_and_gps_privacy(tmp_path):
    p = _jpeg(tmp_path / "camera.jpg", exif=_exif_with_gps())
    md = extract_metadata(p)
    assert md["exif_present"] and md["gps_metadata_present"]
    assert md["camera_make"] == "TestMake" and md["camera_model"] == "TestModel 1"
    assert md["capture_datetime"] == "2024:05:06 07:08:09" and md["iso"] == "400"
    assert md["exposure_time"] == "1/250 s" and md["f_number"] == "f/2.8" and md["focal_length"] == "35 mm"
    flat = " ".join(str(v) for v in md.values())
    for secret in ("56.78", "130.0", "(12.0", "Private Person", "Copyright Owner", "SERIAL123"):
        assert secret not in flat, secret


def test_orientation_preview_keeps_stored_pixels(tmp_path):
    p = _jpeg(tmp_path / "rotated.jpg", size=(80, 40), exif=_exif_with_gps(orientation=6))
    md = extract_metadata(p)
    assert md["orientation_tag"] == "6" and "stored orientation" in orientation_note(md)
    prev = make_preview(p, tmp_path / "prev")
    with Image.open(prev) as im:
        assert im.size == (80, 40)            # not rotated: same orientation the model sees
        assert len(im.getexif()) == 0         # no EXIF left for a browser to re-rotate


def test_rgba_png_and_greyscale_accepted(tmp_path):
    Image.new("RGBA", (32, 32), (1, 2, 3, 0)).save(tmp_path / "a.png")
    Image.new("L", (32, 32), 7).save(tmp_path / "g.jpeg")
    Image.new("RGB", (32, 32)).save(tmp_path / "w.webp")
    for f in ("a.png", "g.jpeg", "w.webp"):
        assert validate_image(tmp_path / f)[0], f
    assert extract_metadata(tmp_path / "a.png")["colour_mode"] == "RGBA"


def test_uppercase_extension_accepted(tmp_path):
    assert validate_image(_jpeg(tmp_path / "PHOTO.JPG"))[0]


def test_unsupported_text_file(tmp_path):
    p = tmp_path / "notes.txt"; p.write_text("not an image")
    ok, st, note = validate_image(p)
    assert (ok, st) == (False, "UNSUPPORTED") and "not supported" in note


def test_unsupported_image_type(tmp_path):
    p = tmp_path / "scan.tif"; Image.new("RGB", (32, 32)).save(p)
    assert validate_image(p)[:2] == (False, "UNSUPPORTED")   # the official dataset would not even read it


def test_text_renamed_as_jpeg_fails(tmp_path):
    p = tmp_path / "fake.jpg"; p.write_bytes(b"hello, I am not a JPEG")
    assert validate_image(p)[:2] == (False, "FAILED")
    extract_metadata(p)   # must not raise


def test_truncated_jpeg_fails_even_though_model_module_allows_truncation(tmp_path):
    from PIL import ImageFile
    full = _jpeg(tmp_path / "full.jpg", size=(400, 300))
    data = full.read_bytes()
    p = tmp_path / "truncated.jpg"; p.write_bytes(data[: len(data) // 2])
    ImageFile.LOAD_TRUNCATED_IMAGES = True       # what data/datasets.py sets globally
    try:
        assert validate_image(p)[:2] == (False, "FAILED")
        assert ImageFile.LOAD_TRUNCATED_IMAGES is True   # restored afterwards
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def test_empty_and_tiny(tmp_path):
    e = tmp_path / "empty.png"; e.write_bytes(b"")
    assert validate_image(e)[:2] == (False, "FAILED")
    t = tmp_path / "tiny.png"; Image.new("RGB", (4, 4)).save(t)
    assert validate_image(t)[:2] == (False, "UNSUPPORTED")


def test_validation_does_not_modify_file(tmp_path):
    p = _jpeg(tmp_path / "x.jpg", exif=_exif_with_gps())
    before = p.read_bytes()
    validate_image(p); extract_metadata(p); make_preview(p, tmp_path / "pv")
    assert p.read_bytes() == before


def test_png_generator_text_chunk_reported_by_key_only(tmp_path):
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo(); info.add_text("parameters", "a secret prompt text")
    p = tmp_path / "gen.png"; Image.new("RGB", (32, 32)).save(p, pnginfo=info)
    md = extract_metadata(p)
    assert "parameters" in md["other_embedded_metadata"] and "secret prompt" not in md["other_embedded_metadata"]
