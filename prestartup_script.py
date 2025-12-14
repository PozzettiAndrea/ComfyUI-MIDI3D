# MIT License
# Copyright (c) 2025 ComfyUI-MIDI3D Contributors

"""
MIDI-3D PreStartup Script
Copies example assets to ComfyUI input folder on startup.
"""
import os
import shutil


def copy_example_assets():
    """Copy all files and folders from assets/ directory to ComfyUI input directory."""
    try:
        import folder_paths

        input_folder = folder_paths.get_input_directory()
        custom_node_dir = os.path.dirname(os.path.abspath(__file__))

        # Copy entire assets/ folder structure
        assets_folder = os.path.join(custom_node_dir, "assets")
        if not os.path.exists(assets_folder):
            print(f"[MIDI-3D] Warning: assets folder not found at {assets_folder}")
            return

        copied_count = 0
        for root, dirs, files in os.walk(assets_folder):
            rel_path = os.path.relpath(root, assets_folder)

            if rel_path != '.':
                dest_dir = os.path.join(input_folder, rel_path)
                os.makedirs(dest_dir, exist_ok=True)
            else:
                dest_dir = input_folder

            for file in files:
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)

                if not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    copied_count += 1
                    rel_dest = os.path.join(rel_path, file) if rel_path != '.' else file
                    print(f"[MIDI-3D] Copied {rel_dest} to input/")

        if copied_count > 0:
            print(f"[MIDI-3D] [OK] Copied {copied_count} asset(s) to {input_folder}")
        else:
            print(f"[MIDI-3D] All assets already exist in {input_folder}")

    except Exception as e:
        print(f"[MIDI-3D] Error copying assets: {e}")


# Run on import
copy_example_assets()
