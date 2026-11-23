# Self-motion flow

Learn optic flow from scenes where only the observer moves. Test whether a model detects an independently moving object when camera motion and scene geometry are held fixed.

This synthetic dataset includes camera translation and rotation in rooms and open-floor worlds. Each matched test scene shows the same target absent, stationary, or moving. A stationary target can produce a strong flow boundary from depth alone, making it a useful control for object-motion detection.

## Try a matched scene

[Download the sample (.npz)](sample/matched_scene.npz) or clone this repository and run the example with Python and NumPy.

```sh
python -m pip install numpy
python example.py
```

The script verifies the download, loads all three conditions, and prints target area and estimated-versus-reference flow error. The example is the first scene and first interval in the expanded test manifest. The full dataset download is pending.

## Data

| Split | Flow fields | Independent parent scenes | Resolution |
| --- | ---: | ---: | ---: |
| Training | 100,000 | 20,000 | 128 × 128 |
| Validation | 1,000 | 200 | 128 × 128 |
| Original matched test | 1,800 | 120 | 64 × 64 |
| Expanded matched test | 3,600 | 40 | 128 × 128 |

Training and validation contain stationary worlds. Splits are disjoint by parent scene. In the original test there are 600 fields per target condition. In the expanded test there are 1,200 per condition. Repeated intervals and matched conditions share a parent. Use the scene as the resampling unit when estimating uncertainty.

## Sample format

The first axis follows the condition order in [metadata.json](sample/metadata.json). All depths correspond to the start of the flow interval.

| Array | Shape | Meaning |
| --- | --- | --- |
| `raw` | 3 × 3 × 128 × 128 | Estimated horizontal and vertical flow velocity in pixels/s, then estimated relative inverse axial depth. Flow uses RAFT and depth uses Depth Anything V2. |
| `reference_raw` | 3 × 3 × 128 × 128 | Renderer-derived flow velocity and relative inverse axial depth in the same units. |
| `occupancy` | 3 × 128 × 128 | Target coverage per pixel, ranging from 0 to 1. The stationary target also has an occupancy mask. |
| `valid` | 3 × 128 × 128 | Boolean renderer-derived validity for the reference flow. |

Positive horizontal flow points right and positive vertical flow points down in image coordinates. Multiply velocity by `dt_seconds` to obtain displacement across the interval. Inverse axial depth is divided by its per-field spatial mean. The arrays are supplied before training-set standardization. The original test uses a different pixel grid, so express velocities on a common grid before comparing their magnitudes across resolutions.
