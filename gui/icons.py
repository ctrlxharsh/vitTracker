"""
gui/icons.py
============
Vector icon generator and Lucide icon loader with automatic Pillow fallback.
Zero emojis, crisp high-DPI rendering.
"""

import customtkinter as ctk
from PIL import Image, ImageDraw


def make_icon(name: str, size=(16, 16), color="#ffffff"):
    """Generates crisp vector-style icons using Pillow."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = w // 2, h // 2

    if name == "target":
        r = 5
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        draw.line([cx - r - 2, cy, cx - 2, cy], fill=color, width=2)
        draw.line([cx + 2, cy, cx + r + 2, cy], fill=color, width=2)
        draw.line([cx, cy - r - 2, cx, cy - 2], fill=color, width=2)
        draw.line([cx, cy + 2, cx, cy + r + 2], fill=color, width=2)
    elif name == "reset":
        draw.arc([2, 2, w - 2, h - 2], start=40, end=320, fill=color, width=2)
        draw.polygon([(w - 4, cy - 4), (w, cy + 2), (w - 6, cy + 2)], fill=color)
    elif name == "pause":
        draw.rectangle([4, 3, 7, h - 3], fill=color)
        draw.rectangle([w - 7, 3, w - 4, h - 3], fill=color)
    elif name == "play":
        draw.polygon([(5, 3), (w - 3, cy), (5, h - 3)], fill=color)
    elif name == "camera":
        draw.rounded_rectangle([2, 4, w - 2, h - 3], radius=2, outline=color, width=2)
        draw.ellipse([cx - 3, cy, cx + 3, cy + 5], outline=color, width=2)
        draw.rectangle([cx - 2, 2, cx + 2, 4], fill=color)
    elif name == "folder":
        draw.rectangle([2, 5, w - 2, h - 3], outline=color, width=2)
        draw.line([2, 5, 6, 5, 8, 7, w - 2, 7], fill=color, width=2)
    elif name == "servo":
        # Icon for pan/tilt servo
        draw.rounded_rectangle([3, 4, w - 3, h - 4], radius=2, outline=color, width=2)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=color)
        draw.line([cx, 2, cx, 4], fill=color, width=2)

    return ctk.CTkImage(light_image=img, dark_image=img, size=size)


def get_icon(name: str, size=(16, 16), color="#ffffff"):
    """
    Returns a High-DPI CTkImage using Lucide icons from tkinter-icons,
    with an automatic fallback to native Pillow vector drawing.
    """
    try:
        from tkinter_icons import LucideIcon

        lucide_map = {
            "target": "crosshair",
            "reset": "rotate-ccw",
            "pause": "pause",
            "play": "play",
            "camera": "camera",
            "folder": "folder",
            "servo": "compass",
        }
        icon_name = lucide_map.get(name, name)
        icon = LucideIcon(icon_name, size=size[0], color=color)
        pil_img = icon.to_pil()
        return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)
    except Exception:
        return make_icon(name, size, color)
