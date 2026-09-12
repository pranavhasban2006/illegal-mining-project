import sys, json
sys.path.insert(0, "../../src")
import numpy as np
from PIL import Image
from mining_detector import real_run, preprocess, config, hard_cases

bands_t0_full, transform, crs = preprocess.load_band_stack(real_run._band_paths("../raw/t0_scene_2025"))
bands_t0, aoi_transform = preprocess.crop_to_bbox(bands_t0_full, transform, crs, config.PILOT_AOI_BBOX)

chips = json.load(open("chips_data.json"))
import os
os.makedirs("t0_chips", exist_ok=True)

for c in chips:
    chip = hard_cases.extract_rgb_chip(bands_t0, c["row"], c["col"], chip_size=64)
    Image.fromarray(chip).save(f"t0_chips/{c['chip_id']}.png")

print(f"Extracted {len(chips)} paired t0 chips to t0_chips/")
