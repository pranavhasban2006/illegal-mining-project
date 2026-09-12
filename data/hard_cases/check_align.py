import sys
sys.path.insert(0, "../../src")
from mining_detector import real_run, preprocess, config

bands_t0_full, transform, crs = preprocess.load_band_stack(real_run._band_paths("../raw/t0_scene_2025"))
bands_t1_full, _, _ = preprocess.load_band_stack(real_run._band_paths("../raw/test_scene"))
bands_t0, aoi_transform_t0 = preprocess.crop_to_bbox(bands_t0_full, transform, crs, config.PILOT_AOI_BBOX)
bands_t1, aoi_transform_t1 = preprocess.crop_to_bbox(bands_t1_full, transform, crs, config.PILOT_AOI_BBOX)
print("t0 shape:", bands_t0.shape, "t1 shape:", bands_t1.shape)
print("t0 transform:", aoi_transform_t0)
print("t1 transform:", aoi_transform_t1)
print("Shapes match:", bands_t0.shape == bands_t1.shape)
