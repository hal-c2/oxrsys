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

// Byte 30 says what a packet carries.
constexpr size_t kKindOffset = 30;
constexpr uint8_t kMagnetometerKind = 0x04;
// Motion packets fill the magnetometer slots with this.
constexpr float kNoMagnetometerReading = -3200.0f;

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

// Maps magnetometer axes to raw gyro axes. Found by turning the glasses
// through many orientations: this is the mapping under which the field stays
// fixed in the world while the gyro says the glasses turn.
const glm::mat3 kMagnetometerToRaw = glm::mat3(glm::vec3(0.0f, 0.0f, -1.0f),  // mag X -> raw -Z
                                               glm::vec3(1.0f, 0.0f, 0.0f),   // mag Y -> raw X
                                               glm::vec3(0.0f, -1.0f, 0.0f)); // mag Z -> raw -Y

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
// Magnetometer: a window of this long, turned at least this much, adds one
// hard-iron equation; old ones fade, and the starting offset weighs as a few.
constexpr uint64_t kMagWindowUs = 500000;
constexpr float kMagWindowMinTurn = 0.1f;  // rad
constexpr float kMagFade = 0.995f;
constexpr float kMagPriorWeight = 0.05f;
// Yaw is pulled toward the start heading at this rate (1/s), and the gyro's
// yaw bias learned from what is left, from readings whose strength is within
// kMagDisturbance of usual and not near vertical.
constexpr float kHeadingGain = 0.3f;
constexpr float kHeadingBiasGain = 0.02f;
constexpr float kMagDisturbance = 0.2f;
constexpr float kMagMinHorizontal = 0.2f;

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

float WrapAngle(float angle)
{
    return std::remainder(angle, 2.0f * glm::pi<float>());
}

// Heading of a world vector, measured like YawOf: 0 toward -Z, positive left.
float HeadingOf(const glm::vec3& v)
{
    return std::atan2(-v.x, -v.z);
}

float YawOf(const glm::quat& q)
{
    const glm::vec3 forward = q * glm::vec3(0.0f, 0.0f, -1.0f);
    const float yaw = std::atan2(-forward.x, -forward.z);
    return std::isfinite(yaw) ? yaw : 0.0f;
}

} // namespace

XrealImu::XrealImu(std::string address)
    : XrealImu(std::move(address), Options{})
{
}

XrealImu::XrealImu(std::string address, const Options& options)
    : address_(std::move(address)),
      options_(options),
      mountCorrection_(glm::mat3_cast(glm::angleAxis(glm::radians(options.pitchOffsetDeg), glm::vec3(1.0f, 0.0f, 0.0f))))
{
    magOffsetPrior_ = mountCorrection_ * kRawToHead * kMagnetometerToRaw * options.magnetometerOffset;
    magOffset_ = magOffsetPrior_;
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
    if (data[kKindOffset] == kMagnetometerKind)
    {
        const glm::vec3 raw(ReadF32(data + 58), ReadF32(data + 62), ReadF32(data + 66));
        if (!IsFinite(raw) || raw.x == kNoMagnetometerReading || glm::length(raw) > 5000.0f)
        {
            return false;
        }
        sample.kind = Sample::Kind::Magnetometer;
        sample.timestampUs = ReadU64(data + 14) / 1000;
        sample.magnetometer = kRawToHead * kMagnetometerToRaw * raw;
        return true;
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
    Sample sample = rawSample;
    sample.gyro = mountCorrection_ * rawSample.gyro;
    sample.accel = mountCorrection_ * rawSample.accel;
    sample.magnetometer = mountCorrection_ * rawSample.magnetometer;
    std::scoped_lock lock(mutex_);
    if (sample.kind == Sample::Kind::Magnetometer)
    {
        IntegrateMagnetometer(sample);
    }
    else
    {
        IntegrateMotion(sample);
    }
}

void XrealImu::IntegrateMotion(const Sample& sample)
{
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
    magWindowTurn_ = IntegrateRate(magWindowTurn_, rate, dt);

    // Tilt correction: steer the estimated up vector toward the measured one.
    if (accelTrusted)
    {
        const glm::vec3 measuredUp = sample.accel / accelMagnitude;
        const glm::vec3 estimatedUp = glm::inverse(orientation_) * glm::vec3(0.0f, 1.0f, 0.0f);
        rate += glm::cross(measuredUp, estimatedUp) * (startup ? kTiltGainStartup : kTiltGain);
    }

    orientation_ = IntegrateRate(orientation_, rate, dt);
}

void XrealImu::IntegrateMagnetometer(const Sample& sample)
{
    if (!options_.magnetometer || !hasOrientation_)
    {
        return;
    }
    const float dt = static_cast<float>(sample.timestampUs - lastMagUs_) * 1e-6f;
    lastMagUs_ = sample.timestampUs;

    // Hard iron: over a window the head turned by `turn` (head axes), a fixed
    // world field reads field_b = turn^T field_a, so with raw = field + offset,
    // raw_b - turn^T raw_a = (I - turn^T) offset: one linear equation per window.
    if (!magWindowStarted_)
    {
        magWindowStarted_ = true;
        magWindowField_ = sample.magnetometer;
        magWindowTurn_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        magWindowStartUs_ = sample.timestampUs;
    }
    else if (sample.timestampUs - magWindowStartUs_ >= kMagWindowUs)
    {
        const glm::quat turn = magWindowTurn_;
        if (2.0f * std::atan2(glm::length(glm::vec3(turn.x, turn.y, turn.z)), std::abs(turn.w)) > kMagWindowMinTurn)
        {
            const glm::mat3 turnT = glm::transpose(glm::mat3_cast(turn));
            const glm::mat3 m = glm::mat3(1.0f) - turnT;
            const glm::vec3 y = sample.magnetometer - turnT * magWindowField_;
            magNormal_ = magNormal_ * kMagFade + glm::transpose(m) * m;
            magRhs_ = magRhs_ * kMagFade + glm::transpose(m) * y;
            magOffset_ = glm::inverse(magNormal_ + glm::mat3(kMagPriorWeight)) *
                         (magRhs_ + kMagPriorWeight * magOffsetPrior_);
        }
        magWindowField_ = sample.magnetometer;
        magWindowTurn_ = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        magWindowStartUs_ = sample.timestampUs;
    }

    // Heading: hold yaw to where the field pointed at start, skipping readings
    // a nearby magnet disturbs (strength off) or that say little about heading.
    const glm::vec3 field = sample.magnetometer - magOffset_;
    const float strength = glm::length(field);
    magStrength_ = magStrength_ == 0.0f ? strength : magStrength_ + (strength - magStrength_) * 0.002f;
    const glm::vec3 world = orientation_ * field;
    if (std::abs(strength - magStrength_) > kMagDisturbance * magStrength_ ||
        glm::length(glm::vec2(world.x, world.z)) < kMagMinHorizontal * strength)
    {
        return;
    }
    const float heading = HeadingOf(world);
    // Signed: a magnetometer packet can be stamped a little before the newest motion one.
    const bool settled = static_cast<float>(static_cast<int64_t>(sample.timestampUs - firstTimestampUs_)) * 1e-6f >=
                         kStartupSeconds;
    if (!hasMagHeading_)
    {
        if (settled)
        {
            magHeading_ = heading;
            hasMagHeading_ = true;
        }
        return;
    }
    if (!(dt > 0.0f) || dt > 0.05f)
    {
        return;
    }
    // Heading too far left means the gyro reads left turns too high: turn back,
    // and move its bias toward world up (in head axes).
    const float error = WrapAngle(heading - magHeading_);
    orientation_ = glm::normalize(glm::angleAxis(-error * kHeadingGain * dt, glm::vec3(0.0f, 1.0f, 0.0f)) * orientation_);
    gyroBias_ += glm::inverse(orientation_) * glm::vec3(0.0f, error * kHeadingBiasGain * dt, 0.0f);
}

glm::vec3 XrealImu::GetMagnetometerOffset() const
{
    std::scoped_lock lock(mutex_);
    return magOffset_;
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
