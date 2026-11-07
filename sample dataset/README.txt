FY-2H SVISSR - 9-CHANNEL MODEL DATASET  (SAMPLE)
==================================================

SOURCE
  Satellite : FY-2H (Fengyun-2H), Chinese geostationary weather satellite @ 79 deg E
  Instrument: SVISSR (Stretched Visible & Infrared Spin-Scan Radiometer)
  Imagery   : longwave-infrared base (IR1), received & decoded locally via SatDump
  Region    : Indian subcontinent  (approx 62.3-93.9 deg E, 5.3-34.3 deg N)

FULL DATASET (available on request, ~57 GB)
  Period    : 1 Feb 2026 -> 20 May 2026  (109 days)
  Frames    : 4,360 files, ~30-min cadence (~40 frames/day)

THIS SAMPLE
  6 authentic, unmodified frames spanning the period:
    20260201_0530   20260215_1229   20260315_0830
    20260415_1529   20260515_0329   20260520_0029
  + norm_stats.json
  + preview_IR1_contact_sheet.png
  (frames are in the zip: FY2H_NPY9CH_sample.zip)

ARRAY FORMAT
  Each .npy : shape (589, 651, 9), dtype float32
  Grid      : ~5 km / pixel (FY-2 SVISSR IR nominal)
  Filename  : YYYYMMDD_HHMM  (UTC)

  Channel order (index -> name):
    0  ir1                         10.3-11.3 um  LWIR window (base imagery)
    1  ir2                         11.5-12.5 um  LWIR split-window
    2  ir3                         3.5-4.0  um   mid-wave IR
    3  wv                          6.3-7.6  um   water vapour
    4  vis                         0.55-0.99 um  visible
    5  split_window_ir1_minus_ir2  derived
    6  wv_minus_ir1                derived
    7  temporal_delta_ir1          derived (IR1 change vs previous frame)
    8  sobel_ir1                   derived (edge/gradient of IR1)

  Values are scaled to ~[0,1] per channel. norm_stats.json holds the per-channel
  mean/std used for training-time standardization.

QUICK START (Python)
  import numpy as np
  a = np.load("frames/20260201_0530.npy")   # (589, 651, 9) float32
  ir1 = a[:, :, 0]                            # base LWIR channel
  # view as 8-bit image:
  img = np.clip(ir1 * 255, 0, 255).astype("uint8")

Contact: nitishksn24@gmail.com
