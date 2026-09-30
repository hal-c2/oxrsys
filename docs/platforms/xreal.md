# XREAL Local Display

## Scope

Local display mode runs OpenXR apps directly on XREAL One / One Pro glasses plugged into the Mac, without a streaming client. The runtime presents the submitted eye images on the glasses' screen and reads 3DoF head orientation from the glasses' IMU. It is macOS-only (AppKit + Metal); other platforms build a no-op presenter.

Tested with an XREAL One Pro and Qt Quick 3D XR (Qt 6.11, Metal backend).

## Setup

1. Connect the glasses over USB-C. macOS exposes them as a display and as a USB network interface; the IMU stream is at `169.254.2.1:52998`.
2. In System Settings > Displays, set the glasses to extend the desktop (not mirror).
3. On the glasses, pick the display mode where the screen follows your head (not the spatially anchored "fixed" mode). The runtime compensates head rotation itself; an anchored screen would compensate twice.
4. For stereo, enable the glasses' 3D side-by-side mode. macOS then reports a 3840x1080 screen and the runtime shows one eye per half. On a 16:9 screen the runtime shows the left eye only. "Real 3D" (the glasses' 2D-to-3D conversion) is not needed; apps already render true stereo.
5. Enable the mode in `~/Library/Application Support/OXRSys/oxrsys-runtime.toml`:

   ```toml
   local_display_enabled = true
   ```

6. Launch the app with `XR_RUNTIME_JSON` pointing at `build/runtime/oxrsys-runtime.json`.

The streaming server is not started in this mode.

## Behavior

- Screen selection: the first screen whose name contains `local_display_screen` (default `XREAL`). Without a match, a 1280x360 preview window opens on the main screen.
- Frame pacing follows the target screen's refresh rate (120 Hz mirrored/2D, 60 Hz in side-by-side mode on the One Pro).
- Recommended per-eye size and FOV default to the One Pro panels and optics: 1920x1080, 50.6 x 29.8 degrees.
- Head tracking: the IMU (1 kHz) is fused with a complementary filter; gyro bias is learned while the head is still. Yaw starts at zero and `LOCAL` space is anchored at the first tracked pose. Position is fixed (3DoF).
- Yaw drift: the glasses also send magnetometer packets (400 Hz, a separate packet kind whose gyro and accelerometer are NaN). Yaw is pulled gently toward the magnetic heading it had at start, ignoring readings whose strength says a magnet is near. The glasses' own hard-iron offset (several times Earth's field) starts from a One Pro measurement and is refined by least squares while the head turns: over each short turn a fixed world field must rotate exactly as the gyro says.
- Timewarp: before presenting, each eye is re-projected by the rotation between the render pose and the newest predicted IMU pose.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `local_display_enabled` | `false` | Present locally instead of streaming |
| `local_display_screen` | `"XREAL"` | Substring of the target screen name |
| `local_display_eye_width` / `_height` | `1920` / `1080` | Recommended per-eye render size |
| `local_display_fov_h_deg` / `_v_deg` | `50.6` / `29.8` | Symmetric per-eye FOV |
| `local_display_ipd_mm` | `63.0` | Eye separation |
| `local_display_timewarp` | `true` | Present-time rotational reprojection |
| `local_display_gamma_encoded_sources` | `true` | sRGB swapchains hold display-ready values (Qt Quick 3D XR); turn off if an engine's colors look too dark |
| `local_display_render_prediction_ms` | `16` | Pose prediction for `xrLocateViews` |
| `local_display_warp_prediction_ms` | `8` | Pose prediction at present time |
| `xreal_imu_address` | `"169.254.2.1:52998"` | IMU TCP endpoint |
| `xreal_imu_pitch_offset_deg` | `0` | Horizon fine-tuning; positive lowers the view |
| `xreal_magnetometer` | `true` | Hold yaw to the magnetic heading at start (no sideways drift) |
| `xreal_magnetometer_offset` | `""` | Starting hard-iron offset `"x, y, z"` (uT, magnetometer axes); empty uses the One Pro's |

## IMU Calibration

The raw-to-head axis mapping in `runtime/src/XrealImu.cpp` was measured on a One Pro: yaw-left rotates about raw -Y, pitch-up about raw +X, and the IMU sits pitched about 38 degrees relative to the displays. The magnetometer's axes differ from the gyro's (`kMagnetometerToRaw`), found as the mapping under which the field stays fixed in the world while the glasses turn. Other One-series models may need a different mapping or `xreal_imu_pitch_offset_deg`.
