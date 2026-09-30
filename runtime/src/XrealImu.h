// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// 3DoF head tracker for XREAL One / One Pro glasses.
//
// The glasses expose their IMU as a TCP stream on their USB network interface
// (169.254.2.1:52998 by default). Packets are parsed the same way as the
// xreal_one_driver crate, then gyro and accelerometer are fused (Mahony-style
// complementary filter with at-rest gyro bias estimation) into a gravity-aligned
// orientation in OpenXR axes: +X right, +Y up, -Z forward.
class XrealImu
{
public:
    struct Sample
    {
        uint64_t timestampUs = 0;
        glm::vec3 gyro = {};  // rad/s, head frame, OpenXR axes
        glm::vec3 accel = {}; // m/s^2 specific force, head frame, OpenXR axes
    };

    // pitchOffsetDeg fine-tunes the IMU-to-display pitch on top of the built-in
    // calibration; positive values lower the rendered view.
    explicit XrealImu(std::string address, float pitchOffsetDeg = 0.0f);
    ~XrealImu();

    XrealImu(const XrealImu&) = delete;
    XrealImu& operator=(const XrealImu&) = delete;

    void Start();
    void Stop();

    bool HasOrientation() const;

    // Head-to-world orientation, extrapolated predictionSeconds past the newest sample.
    glm::quat GetOrientation(float predictionSeconds = 0.0f) const;

    // Re-zero yaw so the current forward direction becomes -Z.
    void RecenterYaw();

    // Exposed for tests. TryParsePacket reads one packet starting at data[0];
    // ExtractSamples consumes every complete packet from the front of buffer.
    static constexpr size_t PacketSize = 84;
    static bool TryParsePacket(const uint8_t* data, size_t size, Sample& sample);
    static void ExtractSamples(std::vector<uint8_t>& buffer, std::vector<Sample>& samples);

    // Pure fusion step, exposed for tests.
    void Integrate(const Sample& sample);

private:
    void Run();
    int Connect() const;

    std::string address_;
    glm::mat3 mountCorrection_;
    std::atomic_bool running_{false};
    std::thread thread_;

    mutable std::mutex mutex_;
    bool hasOrientation_ = false;
    glm::quat orientation_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // head -> world
    glm::quat yawOffset_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    glm::vec3 angularVelocity_ = {};                             // bias-corrected, head frame
    glm::vec3 gyroBias_ = {};
    glm::vec3 gyroLowPass_ = {};
    uint64_t lastTimestampUs_ = 0;
    uint64_t firstTimestampUs_ = 0;
    uint32_t stillSampleCount_ = 0;
};
