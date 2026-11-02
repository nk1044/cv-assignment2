"""Polygon mask annotator for the captured photos (OpenCV window).

    python -m src.annotate [name ...]     # default: every photo in captured/images_256

The 256x256 photo is shown enlarged with the current mask tinted green.
    left click    add a polygon vertex
    right click   close the polygon and add it to the mask (or cut it out, in cut mode)
    x             toggle add / cut mode (cut removes regions, e.g. the hole in a mug handle)
    u             undo the last vertex       c   clear the whole mask
    s             save and go to the next photo
    n             skip this photo            q   quit
Masks are saved to captured/masks_256/<name>.png with object 255 and background 0. An
existing mask is loaded so you can keep editing it.
"""
import sys

import cv2
import numpy as np

from src.config import CAPTURED, SIZE

ZOOM = 3
SUBPIXEL_BITS = 4  # cv2.fillPoly fractional precision, so clicks between pixels are respected


def annotate(name):
    img = cv2.imread(str(CAPTURED / "images_256" / f"{name}.png"), cv2.IMREAD_COLOR)
    mask_path = CAPTURED / "masks_256" / f"{name}.png"
    old = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (old >= 128).astype(np.uint8) * 255 if old is not None else np.zeros((SIZE, SIZE), np.uint8)
    state = {"pts": [], "cut": False}

    def close_polygon():
        if len(state["pts"]) >= 3:
            pts = np.round(np.array(state["pts"]) * (1 << SUBPIXEL_BITS)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 0 if state["cut"] else 255, cv2.LINE_8, SUBPIXEL_BITS)
        state["pts"] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pts"].append(((x + 0.5) / ZOOM - 0.5, (y + 0.5) / ZOOM - 0.5))
        elif event == cv2.EVENT_RBUTTONDOWN:
            close_polygon()

    window = f"annotate {name}"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        view = img.copy()
        view[mask > 0] = (0.55 * view[mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        view = cv2.resize(view, (SIZE * ZOOM, SIZE * ZOOM), interpolation=cv2.INTER_NEAREST)
        colour = (0, 0, 255) if state["cut"] else (255, 255, 0)
        pts = [(int((px + 0.5) * ZOOM), int((py + 0.5) * ZOOM)) for px, py in state["pts"]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(view, a, b, colour, 1)
        for pt in pts:
            cv2.circle(view, pt, 3, colour, -1)
        cv2.putText(view, "CUT" if state["cut"] else "ADD", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow(window, view)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("x"):
            state["cut"] = not state["cut"]
        elif key == ord("u") and state["pts"]:
            state["pts"].pop()
        elif key == ord("c"):
            mask[:] = 0
            state["pts"] = []
        elif key == ord("s"):
            close_polygon()
            cv2.imwrite(str(mask_path), mask)
            print(f"saved {mask_path}")
            break
        elif key in (ord("n"), ord("q")):
            break
    cv2.destroyWindow(window)
    return key != ord("q")


def main():
    (CAPTURED / "masks_256").mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or sorted(p.stem for p in (CAPTURED / "images_256").glob("*.png"))
    for name in names:
        if not annotate(name):
            break


if __name__ == "__main__":
    main()
