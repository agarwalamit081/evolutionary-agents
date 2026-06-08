"""Check WCAG 2.1 color contrast ratio between two hex colors.
Usage: python check_contrast.py <foreground_hex> <background_hex>
Example: python check_contrast.py #71717A #FFFFFF
"""
import re
import sys


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Use 6-digit format (e.g., #FFFFFF)")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_luminance(rgb):
    def adjust(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = [adjust(c) for c in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def check_contrast(fg_hex, bg_hex):
    fg_rgb = hex_to_rgb(fg_hex)
    bg_rgb = hex_to_rgb(bg_hex)
    lighter = max(get_luminance(fg_rgb), get_luminance(bg_rgb))
    darker = min(get_luminance(fg_rgb), get_luminance(bg_rgb))
    ratio = (lighter + 0.05) / (darker + 0.05)

    print(f"Foreground: {fg_hex} | Background: {bg_hex}")
    print(f"Contrast Ratio: {ratio:.2f}:1")
    if ratio >= 4.5:
        print("PASS: Meets WCAG AA for normal text.")
    elif ratio >= 3.0:
        print("WARNING: Meets WCAG AA for LARGE text only (18pt+ or 14pt+ bold).")
    else:
        print("FAIL: Does not meet WCAG AA standards.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and all(re.match(r"^#[0-9a-fA-F]{6}$", c) for c in sys.argv[1:3]):
        check_contrast(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python check_contrast.py <foreground_hex> <background_hex>")
        print("Example: python check_contrast.py #71717A #FFFFFF")
