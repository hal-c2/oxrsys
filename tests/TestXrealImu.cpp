// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "XrealImu.h"
#include <cmath>
#include <cstring>
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
        imu.Integrate({timestampUs, gyro, accel});
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
