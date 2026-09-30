// SPDX-License-Identifier: MPL-2.0

// Local display mode is macOS-only for now (AppKit + Metal); other platforms get
// a no-op presenter so the runtime still links.

#include "LocalDisplay.h"

#include <spdlog/spdlog.h>

struct LocalDisplay::Impl
{
};

LocalDisplay::LocalDisplay(void*, const Settings&)
    : impl_(std::make_unique<Impl>())
{
    spdlog::error("LocalDisplay: local display mode is not supported on this platform");
}

LocalDisplay::~LocalDisplay() = default;

void LocalDisplay::Present(FrameSource&&, const XrFovf[2], const glm::quat&, const glm::quat&)
{
}

uint32_t LocalDisplay::GetRefreshRateHz() const
{
    return 0;
}
