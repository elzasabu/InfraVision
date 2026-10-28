"""Fusion: put branch B's boxes on branch A's colorized frame."""
import cv2
import numpy as np


def rescale_boxes(boxes, in_size, out_size):
    """Map xyxy boxes from an in_size=(w, h) frame to an out_size=(w, h) frame.

    x is scaled by out_w / in_w and y by out_h / in_h, then clipped to the
    output frame, so the boxes stay on the person when SR changed the size.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    (iw, ih), (ow, oh) = in_size, out_size
    if iw <= 0 or ih <= 0:
        raise ValueError(f"invalid input size {in_size}")
    sx, sy = ow / iw, oh / ih
    out = boxes * np.array([sx, sy, sx, sy], np.float32)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, ow)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, oh)
    return out


def _label(img, text, org, scale, thick, fg, bg):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    y = max(y, th + base + 2)
    cv2.rectangle(img, (x, y - th - base - 2), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def draw_detections(rgb, boxes, scores, box_color=(0, 255, 0), outline_color=(0, 0, 0), thickness=0,
                    show_conf=True, label_people_count=True, **_):
    """Thick high-contrast boxes, confidence labels and a 'People: N' banner. Returns a copy."""
    img = np.ascontiguousarray(rgb.copy())
    h, w = img.shape[:2]
    t = int(thickness) or max(2, round(max(h, w) / 400))
    fs = max(0.5, max(h, w) / 1600)
    color, outline = tuple(int(c) for c in box_color), tuple(int(c) for c in outline_color)
    for (x1, y1, x2, y2), s in zip(np.asarray(boxes).reshape(-1, 4), np.asarray(scores).reshape(-1)):
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        cv2.rectangle(img, p1, p2, outline, t + 2 * max(1, t // 2), cv2.LINE_AA)
        cv2.rectangle(img, p1, p2, color, t, cv2.LINE_AA)
        if show_conf:
            _label(img, f"{float(s):.2f}", (p1[0], p1[1] - t), fs * 0.8, max(1, t // 2), outline, color)
    if label_people_count:
        _label(img, f"People: {len(scores)}", (t * 4, t * 4 + int(30 * fs)), fs * 1.4, max(2, t), (255, 255, 255),
               (200, 0, 0) if len(scores) else (60, 60, 60))
    return img
