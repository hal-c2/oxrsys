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
// (169.254.2.1:52998 by default): motion packets (gyro and accelerometer,
// 1 kHz) and magnetometer packets (400 Hz). Gyro and accelerometer are fused
// (Mahony-style complementary filter with at-rest gyro bias estimation) into a
// gravity-aligned orientation in OpenXR axes: +X right, +Y up, -Z forward. The
// magnetometer holds yaw to the heading it had at start, so it does not drift;
// its hard-iron offset is refined while the head turns.
class XrealImu
{
public:
    struct Sample
    {
        enum class Kind
        {
            Motion,
            Magnetometer,
        };

        Kind kind = Kind::Motion;
        uint64_t timestampUs = 0;
        glm::vec3 gyro = {};         // rad/s, head frame, OpenXR axes
        glm::vec3 accel = {};        // m/s^2 specific force, head frame, OpenXR axes
        glm::vec3 magnetometer = {}; // uT, head frame, before the hard-iron offset
    };

    struct Options
    {
        // Fine-tunes the IMU-to-display pitch on top of the built-in
        // calibration; positive values lower the rendered view.
        float pitchOffsetDeg = 0.0f;
        // Off by default: see docs/platforms/xreal.md (Yaw drift).
        bool magnetometer = false;
        // Starting hard-iron offset in the magnetometer's own axes (uT),
        // measured on an XREAL One Pro; refined while the head turns.
        glm::vec3 magnetometerOffset = {-152.4f, 125.4f, -87.3f};
    };

    explicit XrealImu(std::string address);
    XrealImu(std::string address, const Options& options);
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

    // The current hard-iron estimate, head frame (uT). For tests.
    glm::vec3 GetMagnetometerOffset() const;

private:
    void Run();
    int Connect() const;
    void IntegrateMotion(const Sample& sample);
    void IntegrateMagnetometer(const Sample& sample);

    std::string address_;
    Options options_;
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

    // Magnetometer: the hard-iron estimate and the normal equations behind it,
    // the window it is being fitted over, and the heading yaw is held to.
    glm::vec3 magOffset_ = {};
    glm::vec3 magOffsetPrior_ = {};
    glm::mat3 magNormal_ = glm::mat3(0.0f);
    glm::vec3 magRhs_ = {};
    bool magWindowStarted_ = false;
    glm::vec3 magWindowField_ = {};
    // The window's turn by the gyro alone: the fused orientation carries the
    // magnetometer's own corrections, which would feed back into the fit.
    glm::quat magWindowTurn_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    uint64_t magWindowStartUs_ = 0;
    uint64_t lastMagUs_ = 0;
    float magStrength_ = 0.0f;
    bool hasMagHeading_ = false;
    float magHeading_ = 0.0f;
};
