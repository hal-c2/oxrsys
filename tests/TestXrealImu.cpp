// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "XrealImu.h"
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace
{

constexpr float kG = 9.80665f;

std::vector<uint8_t> MakePacket(uint64_t timestampNs, const float gyro[3], const float accel[3])
{
    std::vector<uint8_t> packet(XrealImu::PacketSize, 0);
    const uint8_t header[6] = {0x28, 0x36, 0x00, 0x00, 0x00, 0x80};
    const uint8_t sensorTag[6] = {0x00, 0x40, 0x1f, 0x00, 0x00, 0x40};
    std::memcpy(packet.data(), header, sizeof(header));
    std::memcpy(packet.data() + 6, sensorTag, sizeof(sensorTag));
    std::memcpy(packet.data() + 14, &timestampNs, sizeof(timestampNs));
    std::memcpy(packet.data() + 34, gyro, 3 * sizeof(float));
    std::memcpy(packet.data() + 46, accel, 3 * sizeof(float));
    return packet;
}

// Feeds `seconds` of 1 kHz samples with a constant head-frame rate and accel.
void Feed(XrealImu& imu, uint64_t& timestampUs, float seconds, glm::vec3 gyro, glm::vec3 accel)
{
    const int count = static_cast<int>(seconds * 1000.0f);
    for (int i = 0; i < count; ++i)
    {
        timestampUs += 1000;
        XrealImu::Sample sample;
        sample.timestampUs = timestampUs;
        sample.gyro = gyro;
        sample.accel = accel;
        imu.Integrate(sample);
    }
}

glm::vec3 Forward(const glm::quat& q)
{
    return q * glm::vec3(0.0f, 0.0f, -1.0f);
}

} // namespace

TEST_CASE("XrealImu parses IMU packets out of a noisy stream", "[xreal]")
{
    const float gyro[3] = {0.1f, -0.2f, 0.3f};
    const float accel[3] = {0.5f, -6.2f, 7.6f};

    std::vector<uint8_t> stream = {0x01, 0x02, 0x28, 0x36}; // junk, including a partial header
    auto first = MakePacket(5'000'000, gyro, accel);
    auto second = MakePacket(6'000'000, gyro, accel);
    stream.insert(stream.end(), first.begin(), first.end());
    stream.insert(stream.end(), second.begin(), second.end());
    stream.insert(stream.end(), second.begin(), second.begin() + 20); // trailing partial packet

    std::vector<XrealImu::Sample> samples;
    XrealImu::ExtractSamples(stream, samples);

    REQUIRE(samples.size() == 2);
    CHECK(samples[0].timestampUs == 5000);
    CHECK(samples[1].timestampUs == 6000);
    CHECK_THAT(glm::length(samples[0].gyro), WithinAbs(std::sqrt(0.14f), 1e-5));
    CHECK_THAT(glm::length(samples[0].accel), WithinAbs(std::sqrt(0.25f + 38.44f + 57.76f), 1e-4));
    CHECK(stream.size() == 20);
}

TEST_CASE("XrealImu rejects packets without the sensor tag or with non-finite data", "[xreal]")
{
    const float gyro[3] = {0.0f, 0.0f, 0.0f};
    const float accel[3] = {0.0f, kG, 0.0f};
    XrealImu::Sample sample;

    auto untagged = MakePacket(1000, gyro, accel);
    std::memset(untagged.data() + 6, 0, 6);
    CHECK_FALSE(XrealImu::TryParsePacket(untagged.data(), untagged.size(), sample));

    const float nanGyro[3] = {NAN, 0.0f, 0.0f};
    auto corrupt = MakePacket(1000, nanGyro, accel);
    CHECK_FALSE(XrealImu::TryParsePacket(corrupt.data(), corrupt.size(), sample));
}

TEST_CASE("XrealImu starts gravity aligned with zero yaw", "[xreal]")
{
    XrealImu imu("127.0.0.1:1");
    uint64_t t = 0;

    // Head pitched down 20 degrees: world up seen from the head is tilted toward +Z.
    const float pitch = glm::radians(-20.0f);
    const glm::vec3 up(0.0f, std::cos(pitch), -std::sin(pitch));
    Feed(imu, t, 0.5f, glm::vec3(0.0f), up * kG);

    REQUIRE(imu.HasOrientation());
    const glm::quat q = imu.GetOrientation();
    const glm::vec3 worldUp = q * up;
    CHECK_THAT(worldUp.y, WithinAbs(1.0, 1e-3));
    const glm::vec3 forward = Forward(q);
    CHECK_THAT(forward.x, WithinAbs(0.0, 1e-3));
    CHECK_THAT(std::asin(forward.y), WithinAbs(pitch, 1e-2));
}

TEST_CASE("XrealImu integrates yaw-left as positive rotation about +Y", "[xreal]")
{
    XrealImu imu("127.0.0.1:1");
    uint64_t t = 0;
    Feed(imu, t, 1.0f, glm::vec3(0.0f), glm::vec3(0.0f, kG, 0.0f));
    Feed(imu, t, 0.5f, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, kG, 0.0f));

    const glm::vec3 forward = Forward(imu.GetOrientation());
    CHECK_THAT(forward.x, WithinAbs(-std::sin(0.5f), 0.01));
    CHECK_THAT(forward.z, WithinAbs(-std::cos(0.5f), 0.01));

    // Prediction extrapolates along the last angular velocity.
    const glm::vec3 predicted = Forward(imu.GetOrientation(0.1f));
    CHECK_THAT(predicted.x, WithinAbs(-std::sin(0.6f), 0.01));

    imu.RecenterYaw();
    CHECK_THAT(Forward(imu.GetOrientation()).z, WithinAbs(-1.0, 1e-4));
}

TEST_CASE("XrealImu learns gyro bias while still so yaw does not drift", "[xreal]")
{
    XrealImu imu("127.0.0.1:1");
    uint64_t t = 0;
    const glm::vec3 bias(0.004f, 0.009f, -0.007f); // typical at-rest bias seen on a One Pro
    Feed(imu, t, 10.0f, bias, glm::vec3(0.0f, kG, 0.0f));

    const glm::vec3 forward = Forward(imu.GetOrientation());
    const float yaw = std::atan2(-forward.x, -forward.z);
    CHECK(std::abs(yaw) < glm::radians(1.0f));
}

namespace
{

float Yaw(const glm::quat& q)
{
    const glm::vec3 forward = Forward(q);
    return std::atan2(-forward.x, -forward.z);
}

// Feeds `seconds` of a head turning as `pose(t)` says: 1 kHz motion samples
// whose gyro reads `gyroBias` too high, and (with a field) 400 Hz magnetometer
// samples of `worldField` plus `hardIron`, all in head axes.
template <typename Pose>
void FeedMotion(XrealImu& imu, float seconds, Pose pose, glm::vec3 gyroBias,
                std::optional<glm::vec3> worldField, glm::vec3 hardIron = {})
{
    const int steps = static_cast<int>(seconds * 1000.0f);
    for (int i = 1; i <= steps; ++i)
    {
        // The body rate from one pose to the next; atan2 keeps the tiny
        // per-millisecond angle exact where acos near 1 would round it away.
        const double t = i * 0.001;
        const glm::quat q = pose(t);
        glm::quat delta = glm::inverse(q) * pose(t + 0.001);
        if (delta.w < 0.0f)
        {
            delta = -delta;
        }
        const glm::vec3 v(delta.x, delta.y, delta.z);
        const float s = glm::length(v);
        const glm::vec3 rate = s > 1e-12f ? v / s * (2.0f * std::atan2(s, delta.w) / 0.001f) : glm::vec3(0.0f);

        XrealImu::Sample motion;
        motion.timestampUs = static_cast<uint64_t>(i) * 1000;
        motion.gyro = rate + gyroBias;
        motion.accel = glm::inverse(q) * glm::vec3(0.0f, kG, 0.0f);
        imu.Integrate(motion);

        if (worldField && i % 5 == 0 && i % 2 == 0)
        {
            XrealImu::Sample magnetometer;
            magnetometer.kind = XrealImu::Sample::Kind::Magnetometer;
            magnetometer.timestampUs = motion.timestampUs;
            magnetometer.magnetometer = glm::inverse(q) * *worldField + hardIron;
            imu.Integrate(magnetometer);
        }
    }
}

// Looking around: yaw sweeping +-40 degrees, pitch +-20, at different rates.
glm::quat LookingAround(double t)
{
    return glm::angleAxis(static_cast<float>(0.7 * std::sin(0.9 * t)), glm::vec3(0.0f, 1.0f, 0.0f)) *
           glm::angleAxis(static_cast<float>(0.35 * std::sin(1.7 * t)), glm::vec3(1.0f, 0.0f, 0.0f));
}

XrealImu::Options NoHardIron()
{
    XrealImu::Options options;
    options.magnetometer = true;
    options.magnetometerOffset = {};
    return options;
}

// Earth's field, 50 uT dipping 60 degrees, pointing north (-Z) at start.
const glm::vec3 kEarthField = glm::vec3(0.0f, -std::sin(glm::radians(60.0f)), -std::cos(glm::radians(60.0f))) * 50.0f;

} // namespace

TEST_CASE("XrealImu parses magnetometer packets and ignores the motion packets' placeholder", "[xreal]")
{
    const float nan3[3] = {NAN, NAN, NAN};
    auto packet = MakePacket(7'000'000, nan3, nan3);
    packet[30] = 0x04;
    const float field[3] = {-114.0f, 131.0f, -117.0f};
    std::memcpy(packet.data() + 58, field, sizeof(field));

    XrealImu::Sample sample;
    REQUIRE(XrealImu::TryParsePacket(packet.data(), packet.size(), sample));
    CHECK(sample.kind == XrealImu::Sample::Kind::Magnetometer);
    CHECK(sample.timestampUs == 7000);
    CHECK_THAT(glm::length(sample.magnetometer), WithinAbs(std::sqrt(114.0f * 114 + 131 * 131 + 117 * 117), 1e-3));

    const float none[3] = {-3200.0f, -3200.0f, -3200.0f};
    std::memcpy(packet.data() + 58, none, sizeof(none));
    CHECK_FALSE(XrealImu::TryParsePacket(packet.data(), packet.size(), sample));
}

TEST_CASE("XrealImu holds yaw to the magnetic heading when the gyro drifts", "[xreal]")
{
    const glm::vec3 bias(0.0f, 0.02f, 0.0f); // 1.1 degrees a second, never still enough to learn

    XrealImu drifting("127.0.0.1:1", NoHardIron());
    FeedMotion(drifting, 30.0f, LookingAround, bias, std::nullopt);
    const float drift = std::abs(Yaw(drifting.GetOrientation()) - Yaw(LookingAround(30.001)));

    XrealImu held("127.0.0.1:1", NoHardIron());
    FeedMotion(held, 30.0f, LookingAround, bias, kEarthField);
    const float error = std::abs(Yaw(held.GetOrientation()) - Yaw(LookingAround(30.001)));

    CHECK(drift > glm::radians(20.0f));
    CHECK(error < glm::radians(5.0f));
}

TEST_CASE("XrealImu learns the glasses' hard-iron offset while the head turns", "[xreal]")
{
    const glm::vec3 hardIron(-40.0f, 25.0f, 60.0f);
    XrealImu imu("127.0.0.1:1", NoHardIron()); // starts assuming none
    FeedMotion(imu, 40.0f, LookingAround, glm::vec3(0.0f), kEarthField, hardIron);

    const glm::vec3 learned = imu.GetMagnetometerOffset();
    CHECK(glm::length(learned - hardIron) < 3.0f);
}
