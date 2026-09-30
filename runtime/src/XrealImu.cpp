// SPDX-License-Identifier: MPL-2.0

#include "XrealImu.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <glm/gtc/constants.hpp>
#include <spdlog/spdlog.h>

namespace
{

constexpr uint8_t kHeader[6] = {0x28, 0x36, 0x00, 0x00, 0x00, 0x80};
constexpr uint8_t kSensorTag[6] = {0x00, 0x40, 0x1f, 0x00, 0x00, 0x40};

constexpr float kGravity = 9.80665f;

// Maps raw sensor axes to OpenXR head axes (+X right, +Y up, -Z forward).
// Determined on an XREAL One Pro by recording yaw-left (rotation about raw -Y),
// pitch-up (raw +X) and right-ear-down roll while worn. The IMU also sits
// pitched ~38 degrees relative to the displays: worn level, it reads 38 degrees
// looking up, so the mapping rotates that back out about +X.
constexpr float kMountPitchDeg = 38.0f;
const glm::mat3 kRawToHead =
    glm::mat3_cast(glm::angleAxis(glm::radians(kMountPitchDeg), glm::vec3(1.0f, 0.0f, 0.0f))) *
    glm::mat3(glm::vec3(1.0f, 0.0f, 0.0f),   // raw X -> right
              glm::vec3(0.0f, -1.0f, 0.0f),  // raw Y -> down
              glm::vec3(0.0f, 0.0f, -1.0f)); // raw Z -> forward

// Filter tuning.
constexpr float kAccelGateMs2 = 1.0f;         // trust accel only when |a| is within this of g
constexpr float kTiltGainStartup = 5.0f;      // fast tilt convergence while settling
constexpr float kTiltGain = 0.5f;             // steady-state tilt correction (rad/s per rad)
constexpr float kStartupSeconds = 2.0f;
constexpr float kStillGyroDeviation = 0.03f;  // rad/s from the low-passed rate
constexpr float kStillMaxRate = 0.1f;         // rad/s, beyond this it is not bias
constexpr uint32_t kStillSamplesRequired = 300;
constexpr float kBiasRateStartup = 0.01f;
constexpr float kBiasRate = 0.001f;

bool IsFinite(const glm::vec3& v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

float ReadF32(const uint8_t* p)
{
    float value = 0.0f;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

uint64_t ReadU64(const uint8_t* p)
{
    uint64_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

glm::quat IntegrateRate(const glm::quat& q, const glm::vec3& rate, float seconds)
{
    const float angle = glm::length(rate) * seconds;
    if (angle < 1e-9f)
    {
        return q;
    }
    return glm::normalize(q * glm::angleAxis(angle, glm::normalize(rate)));
}

// Shortest rotation taking unit vector `from` onto unit vector `to`.
glm::quat RotationBetween(const glm::vec3& from, const glm::vec3& to)
{
    const float d = glm::dot(from, to);
    if (d < -0.9999f)
    {
        return glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
    }
    const glm::vec3 c = glm::cross(from, to);
    return glm::normalize(glm::quat(1.0f + d, c.x, c.y, c.z));
}

float YawOf(const glm::quat& q)
{
    const glm::vec3 forward = q * glm::vec3(0.0f, 0.0f, -1.0f);
    const float yaw = std::atan2(-forward.x, -forward.z);
    return std::isfinite(yaw) ? yaw : 0.0f;
}

} // namespace

XrealImu::XrealImu(std::string address, float pitchOffsetDeg)
    : address_(std::move(address)),
      mountCorrection_(glm::mat3_cast(glm::angleAxis(glm::radians(pitchOffsetDeg), glm::vec3(1.0f, 0.0f, 0.0f))))
{
}

XrealImu::~XrealImu()
{
    Stop();
}

void XrealImu::Start()
{
    if (running_.exchange(true))
    {
        return;
    }
    thread_ = std::thread([this] { Run(); });
}

void XrealImu::Stop()
{
    running_ = false;
    if (thread_.joinable())
    {
        thread_.join();
    }
}

bool XrealImu::TryParsePacket(const uint8_t* data, size_t size, Sample& sample)
{
    if (size < PacketSize || std::memcmp(data, kHeader, sizeof(kHeader)) != 0)
    {
        return false;
    }
    const uint8_t* end = data + PacketSize;
    if (std::search(data, end, std::begin(kSensorTag), std::end(kSensorTag)) == end)
    {
        return false;
    }

    const glm::vec3 rawGyro(ReadF32(data + 34), ReadF32(data + 38), ReadF32(data + 42));
    const glm::vec3 rawAccel(ReadF32(data + 46), ReadF32(data + 50), ReadF32(data + 54));
    if (!IsFinite(rawGyro) || !IsFinite(rawAccel) ||
        glm::length(rawGyro) > 100.0f || glm::length(rawAccel) > 100.0f)
    {
        return false;
    }

    sample.timestampUs = ReadU64(data + 14) / 1000;
    sample.gyro = kRawToHead * rawGyro;
    sample.accel = kRawToHead * rawAccel;
    return true;
}

void XrealImu::ExtractSamples(std::vector<uint8_t>& buffer, std::vector<Sample>& samples)
{
    size_t offset = 0;
    while (buffer.size() - offset >= PacketSize)
    {
        auto begin = buffer.begin() + static_cast<std::ptrdiff_t>(offset);
        auto header = std::search(begin, buffer.end(), std::begin(kHeader), std::end(kHeader));
        if (header == buffer.end())
        {
            // Keep a possible partial header at the tail.
            offset = buffer.size() - std::min(buffer.size() - offset, sizeof(kHeader) - 1);
            break;
        }
        offset = static_cast<size_t>(header - buffer.begin());
        if (buffer.size() - offset < PacketSize)
        {
            break;
        }

        Sample sample;
        if (TryParsePacket(buffer.data() + offset, PacketSize, sample))
        {
            samples.push_back(sample);
            offset += PacketSize;
        }
        else
        {
            // Not an IMU packet (or corrupt); resync past this header.
            offset += sizeof(kHeader);
        }
    }
    buffer.erase(buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(offset));
}

void XrealImu::Integrate(const Sample& rawSample)
{
    const Sample sample = {rawSample.timestampUs, mountCorrection_ * rawSample.gyro,
                           mountCorrection_ * rawSample.accel};
    std::scoped_lock lock(mutex_);

    const float accelMagnitude = glm::length(sample.accel);
    const bool accelTrusted = std::abs(accelMagnitude - kGravity) < kAccelGateMs2;

    if (!hasOrientation_)
    {
        if (!accelTrusted)
        {
            return;
        }
        // Gravity-aligned start with yaw 0: rotate the measured up vector onto +Y.
        const glm::vec3 up = sample.accel / accelMagnitude;
        orientation_ = RotationBetween(up, glm::vec3(0.0f, 1.0f, 0.0f));
        orientation_ = glm::angleAxis(-YawOf(orientation_), glm::vec3(0.0f, 1.0f, 0.0f)) * orientation_;
        gyroLowPass_ = sample.gyro;
        lastTimestampUs_ = sample.timestampUs;
        firstTimestampUs_ = sample.timestampUs;
        hasOrientation_ = true;
        return;
    }

    float dt = static_cast<float>(sample.timestampUs - lastTimestampUs_) * 1e-6f;
    lastTimestampUs_ = sample.timestampUs;
    if (!(dt > 0.0f) || dt > 0.05f)
    {
        return; // clock jump or reconnect; skip rather than integrate garbage
    }
    const bool startup =
        static_cast<float>(sample.timestampUs - firstTimestampUs_) * 1e-6f < kStartupSeconds;

    // Gyro bias: learn only while the head is still (steady rate near zero, accel ~ g).
    gyroLowPass_ += (sample.gyro - gyroLowPass_) * 0.01f;
    const bool still = accelTrusted &&
                       glm::length(sample.gyro - gyroLowPass_) < kStillGyroDeviation &&
                       glm::length(gyroLowPass_) < kStillMaxRate;
    stillSampleCount_ = still ? stillSampleCount_ + 1 : 0;
    if (stillSampleCount_ > kStillSamplesRequired)
    {
        gyroBias_ += (gyroLowPass_ - gyroBias_) * (startup ? kBiasRateStartup : kBiasRate);
    }

    glm::vec3 rate = sample.gyro - gyroBias_;
    angularVelocity_ = rate;

    // Tilt correction: steer the estimated up vector toward the measured one.
    if (accelTrusted)
    {
        const glm::vec3 measuredUp = sample.accel / accelMagnitude;
        const glm::vec3 estimatedUp = glm::inverse(orientation_) * glm::vec3(0.0f, 1.0f, 0.0f);
        rate += glm::cross(measuredUp, estimatedUp) * (startup ? kTiltGainStartup : kTiltGain);
    }

    orientation_ = IntegrateRate(orientation_, rate, dt);
}

bool XrealImu::HasOrientation() const
{
    std::scoped_lock lock(mutex_);
    return hasOrientation_;
}

glm::quat XrealImu::GetOrientation(float predictionSeconds) const
{
    std::scoped_lock lock(mutex_);
    if (!hasOrientation_)
    {
        return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float prediction = std::clamp(predictionSeconds, 0.0f, 0.1f);
    return yawOffset_ * IntegrateRate(orientation_, angularVelocity_, prediction);
}

void XrealImu::RecenterYaw()
{
    std::scoped_lock lock(mutex_);
    yawOffset_ = glm::angleAxis(-YawOf(orientation_), glm::vec3(0.0f, 1.0f, 0.0f));
    spdlog::info("XrealImu: recentered yaw");
}

int XrealImu::Connect() const
{
    const auto colon = address_.rfind(':');
    if (colon == std::string::npos)
    {
        return -1;
    }
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(std::stoi(address_.substr(colon + 1))));
    if (inet_pton(AF_INET, address_.substr(0, colon).c_str(), &addr.sin_addr) != 1)
    {
        return -1;
    }

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        return -1;
    }
    // Non-blocking connect so a missing device times out quickly.
    const int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int result = connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (result != 0 && errno == EINPROGRESS)
    {
        pollfd pfd = {fd, POLLOUT, 0};
        int error = 0;
        socklen_t length = sizeof(error);
        result = (poll(&pfd, 1, 1000) == 1 &&
                  getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) == 0 && error == 0)
            ? 0 : -1;
    }
    if (result != 0)
    {
        close(fd);
        return -1;
    }
    fcntl(fd, F_SETFL, flags);

    timeval timeout = {1, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    int noDelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &noDelay, sizeof(noDelay));
    int noSigPipe = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, sizeof(noSigPipe));
    return fd;
}

void XrealImu::Run()
{
    std::vector<uint8_t> buffer;
    std::vector<Sample> samples;
    uint8_t chunk[4096];
    bool loggedConnectFailure = false;

    while (running_)
    {
        const int fd = Connect();
        if (fd < 0)
        {
            if (!loggedConnectFailure)
            {
                spdlog::warn("XrealImu: cannot reach glasses IMU at {}; retrying", address_);
                loggedConnectFailure = true;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }
        spdlog::info("XrealImu: connected to glasses IMU at {}", address_);
        loggedConnectFailure = false;
        buffer.clear();

        while (running_)
        {
            const ssize_t received = recv(fd, chunk, sizeof(chunk), 0);
            if (received <= 0)
            {
                if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
                {
                    continue;
                }
                spdlog::warn("XrealImu: IMU stream closed; reconnecting");
                break;
            }
            buffer.insert(buffer.end(), chunk, chunk + received);
            samples.clear();
            ExtractSamples(buffer, samples);
            for (const Sample& sample : samples)
            {
                Integrate(sample);
            }
        }
        close(fd);
    }
}
