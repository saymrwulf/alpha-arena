#!/usr/bin/env python3
"""
Generate menu bar icons for Alpha Arena.

Creates template images for macOS menu bar (18x18 @1x, 36x36 @2x).
Uses PIL/Pillow if available, otherwise creates placeholder files.

Usage:
    python scripts/generate-icons.py
"""

import os
from pathlib import Path

# Output directory
ICONS_DIR = Path(__file__).parent.parent / "src" / "macos" / "icons"

# Colors (RGB)
COLOR_GREEN = (34, 197, 94)    # #22C55E - Running
COLOR_GRAY = (100, 116, 139)   # #64748B - Stopped
COLOR_RED = (239, 68, 68)      # #EF4444 - Error

# Icon sizes
SIZE_1X = 18
SIZE_2X = 36


def create_icon_with_pillow(name: str, color: tuple, sizes: list[int]):
    """Create icon using PIL/Pillow."""
    from PIL import Image, ImageDraw

    for size in sizes:
        # Create image with transparency
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw a filled circle
        margin = size // 6
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=color + (255,)  # Add alpha
        )

        # Determine filename
        if size == SIZE_2X:
            filename = f"{name}@2x.png"
        else:
            filename = f"{name}.png"

        # Save
        output_path = ICONS_DIR / filename
        img.save(output_path, 'PNG')
        print(f"Created: {output_path}")


def create_placeholder_icon(name: str, color_name: str):
    """Create placeholder text file when Pillow is not available."""
    # Create a simple 1x1 PNG header as placeholder
    # This is a minimal valid PNG that will display as a single pixel
    # The actual icon will need to be created manually or via another tool

    placeholder_content = f"""# Placeholder for {name}.png
# Color: {color_name}
# Size: 18x18 (plus @2x variant at 36x36)
#
# To create proper icons:
# 1. Install Pillow: pip install Pillow
# 2. Run: python scripts/generate-icons.py
#
# Or create manually:
# - 18x18 PNG with {color_name} filled circle
# - Save as {name}.png and {name}@2x.png (36x36)
"""

    readme_path = ICONS_DIR / f"{name}.txt"
    readme_path.write_text(placeholder_content)
    print(f"Created placeholder: {readme_path}")


def create_template_icons():
    """Create template icons that work in light and dark mode."""
    try:
        from PIL import Image, ImageDraw

        # Template images should be black with alpha for macOS to colorize
        for size in [SIZE_1X, SIZE_2X]:
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            margin = size // 6
            # Black circle with slight transparency
            draw.ellipse(
                [margin, margin, size - margin, size - margin],
                fill=(0, 0, 0, 180)  # Slightly transparent black
            )

            suffix = "@2x" if size == SIZE_2X else ""
            output_path = ICONS_DIR / f"icon_template{suffix}.png"
            img.save(output_path, 'PNG')
            print(f"Created template: {output_path}")

    except ImportError:
        print("Pillow not available for template icons")


def main():
    """Generate all icons."""
    # Ensure directory exists
    ICONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating icons in: {ICONS_DIR}")
    print()

    try:
        from PIL import Image, ImageDraw
        print("Using Pillow to generate icons...")
        print()

        # Generate colored icons
        create_icon_with_pillow("icon_running", COLOR_GREEN, [SIZE_1X, SIZE_2X])
        create_icon_with_pillow("icon_stopped", COLOR_GRAY, [SIZE_1X, SIZE_2X])
        create_icon_with_pillow("icon_error", COLOR_RED, [SIZE_1X, SIZE_2X])

        # Generate template icon (for automatic dark/light mode)
        create_template_icons()

        print()
        print("All icons generated successfully!")

    except ImportError:
        print("Pillow not installed. Creating placeholder files...")
        print("To generate proper icons, run: pip install Pillow")
        print()

        create_placeholder_icon("icon_running", "green (#22C55E)")
        create_placeholder_icon("icon_stopped", "gray (#64748B)")
        create_placeholder_icon("icon_error", "red (#EF4444)")

        print()
        print("Placeholder files created. The menu bar app will use Unicode symbols instead.")


def create_icns():
    """Create .icns app icon (requires iconutil on macOS)."""
    try:
        from PIL import Image, ImageDraw

        # Create iconset directory
        iconset_dir = ICONS_DIR.parent / "Alpha Arena.iconset"
        iconset_dir.mkdir(exist_ok=True)

        # Required sizes for .icns
        sizes = [16, 32, 64, 128, 256, 512, 1024]

        for size in sizes:
            # Create image
            img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Draw filled circle with gradient-like effect
            margin = size // 8
            # Outer circle (darker green)
            draw.ellipse(
                [margin, margin, size - margin, size - margin],
                fill=(22, 163, 74, 255)  # Darker green
            )
            # Inner circle (lighter green)
            inner_margin = size // 5
            draw.ellipse(
                [inner_margin, inner_margin, size - inner_margin, size - inner_margin],
                fill=(34, 197, 94, 255)  # Lighter green
            )

            # Save standard and @2x versions
            img.save(iconset_dir / f"icon_{size}x{size}.png", 'PNG')
            if size <= 512:
                # Create @2x version
                img_2x = img.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
                img_2x.save(iconset_dir / f"icon_{size}x{size}@2x.png", 'PNG')

        print(f"Created iconset at: {iconset_dir}")
        print()
        print("To create .icns file, run:")
        print(f"  iconutil -c icns '{iconset_dir}'")

    except ImportError:
        print("Pillow required for .icns generation")


if __name__ == "__main__":
    import sys

    main()

    # Generate .icns if requested via argument or interactive prompt
    if "--icns" in sys.argv:
        print()
        create_icns()
    elif sys.stdin.isatty():
        print()
        try:
            response = input("Also generate .icns app icon? [y/N]: ")
            if response.lower() == 'y':
                create_icns()
        except EOFError:
            pass
