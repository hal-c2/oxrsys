// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "GraphicsTypes.h"
#include <openxr/openxr.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>

// Presents submitted eye images directly on a local display instead of streaming
// them: a borderless full-screen Metal window on the screen whose name contains
// `screenName` (e.g. XREAL glasses in side-by-side 3D mode). Falls back to a
// preview window on the main screen when no matching screen is attached.
//
// Layout follows the screen shape: a wide (> 2.5:1) screen gets left/right eyes
// side by side; anything else shows the left eye only.
class LocalDisplay
{
public:
    struct Settings
    {
        std::string screenName = "XREAL";
        bool timewarp = true;
        // Treat sRGB swapchain contents as already gamma-encoded. Qt Quick 3D XR
        // writes display-ready values into the sRGB swapchain it picks, which a
        // spec-following present would encode a second time (washed-out colors).
        bool gammaEncodedSources = true;
    };

    LocalDisplay(void* metalDevice, const Settings& settings);
    ~LocalDisplay();

    LocalDisplay(const LocalDisplay&) = delete;
    LocalDisplay& operator=(const LocalDisplay&) = delete;

    // renderOrientation: world head orientation the frame was rendered with.
    // displayOrientation: latest predicted world head orientation at scan-out.
    // With timewarp enabled the image is re-projected by the difference.
    void Present(FrameSource&& frame, const XrFovf fov[2],
                 const glm::quat& renderOrientation, const glm::quat& displayOrientation);

    // Refresh rate of the target screen, 0 until the window exists.
    uint32_t GetRefreshRateHz() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
