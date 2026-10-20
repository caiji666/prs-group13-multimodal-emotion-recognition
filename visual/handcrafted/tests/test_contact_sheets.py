import numpy as np

from m3.contact_sheets import COLS, HEADER, TILE, make_sheet, title_for


def test_sheet_layout_and_missing_face_border():
    frames = np.full((16, 224, 224, 3), 128, np.uint8)
    found = np.ones(16, bool)
    found[5] = False
    sheet = make_sheet(frames, found, np.arange(16), np.arange(16) / 30, "title")
    assert sheet.shape == (HEADER + 4 * TILE, COLS * TILE, 3)
    r, c = divmod(5, COLS)
    top_left = sheet[HEADER + r * TILE + 1, c * TILE + 1]
    assert tuple(top_left) == (0, 0, 255)  # red border (BGR) on the frame without a face


def test_title_decodes_sample_id():
    t = title_for("01-03-02-01-02-12", "train")
    assert "happy" in t and "strong" in t and "actor 12" in t and "train" in t
