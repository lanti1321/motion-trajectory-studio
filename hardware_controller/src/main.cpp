#include <openarm/can/socket/openarm.hpp>
#include <openarm/damiao_motor/dm_motor_constants.hpp>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {
using Arm = openarm::can::socket::OpenArm;
using Motor = openarm::damiao_motor::Motor;
using MIT = openarm::damiao_motor::MITParam;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kJointsPerArm = 8;
constexpr std::size_t kJointCount = 16;
constexpr double kControlHz = 1000.0;
constexpr double kTransitionSeconds = 3.0;
constexpr double kMaxCommandVelocity = 2.0;
constexpr double kMaxCommandAcceleration = 8.0;
constexpr double kHighDynamicsMaxVelocity = 3.5;
constexpr double kHighDynamicsMaxAcceleration = 60.0;
constexpr double kMaxTrackingError = 0.65;
constexpr double kMaxMeasuredVelocity = 6.0;
constexpr int kMaxTemperature = 70;
constexpr auto kHeartbeatTimeout = std::chrono::milliseconds(1500);

constexpr std::array<double, 8> kPositionMin = {
    -2.0943951024, -1.5707963268, -1.5707963268, 0.0,
    -1.5707963268, -1.5707963268, -1.5707963268, -3.1415926536};
constexpr std::array<double, 8> kPositionMax = {
    2.0943951024, 3.1415926536, 1.5707963268, 3.1415926536,
    1.5707963268, 1.5707963268, 1.5707963268, 3.1415926536};
constexpr std::array<double, 8> kKp = {240, 240, 240, 240, 40, 46, 40, 10};
constexpr std::array<double, 8> kKd = {4, 4, 4, 4, .35, .35, .35, .45};

std::atomic_bool running{true};
void signal_handler(int) { running = false; }

struct Frame {
    double time = 0.0;
    std::array<double, kJointCount> q{};
    std::array<double, kJointCount> dq{};
};

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, delimiter)) result.push_back(item);
    return result;
}

class UdpReceiver {
public:
    explicit UdpReceiver(uint16_t port) {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) throw std::runtime_error("cannot create command socket");
        int reuse = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address.sin_port = htons(port);
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0) {
            throw std::runtime_error("cannot bind command port " + std::to_string(port));
        }
        fcntl(fd_, F_SETFL, fcntl(fd_, F_GETFL, 0) | O_NONBLOCK);
    }
    ~UdpReceiver() { if (fd_ >= 0) ::close(fd_); }
    std::vector<std::string> receive_all() {
        std::vector<std::string> messages;
        for (;;) {
            char buffer[8192];
            const auto count = ::recv(fd_, buffer, sizeof(buffer), 0);
            if (count <= 0) break;
            messages.emplace_back(buffer, static_cast<std::size_t>(count));
        }
        return messages;
    }
private:
    int fd_ = -1;
};

class UdpSender {
public:
    explicit UdpSender(bool log_messages = false) : log_messages_(log_messages) {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    }
    ~UdpSender() { if (fd_ >= 0) ::close(fd_); }
    void set_port(uint16_t port) {
        address_ = {};
        address_.sin_family = AF_INET;
        address_.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        address_.sin_port = htons(port);
        enabled_ = port != 0;
    }
    void send(const std::string& value) const {
        if (log_messages_) std::cerr << "[status] " << value << std::endl;
        if (!enabled_) return;
        ::sendto(fd_, value.data(), value.size(), MSG_DONTWAIT,
                 reinterpret_cast<const sockaddr*>(&address_), sizeof(address_));
    }
private:
    int fd_ = -1;
    bool enabled_ = false;
    bool log_messages_ = false;
    sockaddr_in address_{};
};

std::vector<Frame> load_csv(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open trajectory: " + path);
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("trajectory is empty");
    const auto header = split(line, ',');
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;
    if (!columns.count("time")) throw std::runtime_error("trajectory has no time column");
    for (std::size_t i = 0; i < kJointCount; ++i) {
        if (!columns.count("q" + std::to_string(i)))
            throw std::runtime_error("trajectory missing q" + std::to_string(i));
    }
    std::vector<Frame> frames;
    double previous_time = -1.0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto cells = split(line, ',');
        Frame frame;
        auto value = [&](const std::string& name) {
            const auto index = columns.at(name);
            if (index >= cells.size()) throw std::runtime_error("short CSV row");
            const double parsed = std::stod(cells[index]);
            if (!std::isfinite(parsed)) throw std::runtime_error("non-finite CSV value");
            return parsed;
        };
        frame.time = value("time");
        if (frame.time <= previous_time)
            throw std::runtime_error("trajectory time must be strictly increasing");
        previous_time = frame.time;
        for (std::size_t i = 0; i < kJointCount; ++i) {
            frame.q[i] = value("q" + std::to_string(i));
            const auto dq = columns.find("dq" + std::to_string(i));
            frame.dq[i] = dq == columns.end() ? 0.0 : value(dq->first);
            const std::size_t local = i % kJointsPerArm;
            if (frame.q[i] < kPositionMin[local] || frame.q[i] > kPositionMax[local])
                throw std::runtime_error("joint position limit at frame " +
                                         std::to_string(frames.size()) + ", q" +
                                         std::to_string(i));
        }
        frames.push_back(frame);
    }
    if (frames.size() < 2) throw std::runtime_error("trajectory needs at least two frames");
    return frames;
}

std::unique_ptr<Arm> initialize_arm(const std::string& can) {
    using MT = openarm::damiao_motor::MotorType;
    auto arm = std::make_unique<Arm>(can, true);
    arm->init_arm_motors(
        {MT::DM8009, MT::DM8009, MT::DM4340, MT::DM4340,
         MT::DM4310, MT::DM4310, MT::DM4310},
        {1, 2, 3, 4, 5, 6, 7}, {0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17});
    arm->init_gripper_motor(MT::DM4310, 8, 0x18);
    arm->set_callback_mode_all(openarm::damiao_motor::CallbackMode::STATE);
    arm->enable_all();
    for (int i = 0; i < 30; ++i) {
        arm->recv_all(500);
        std::vector<MIT> arm_cmds, grip_cmds;
        for (const auto& motor : arm->get_arm().get_motors())
            arm_cmds.push_back({0.0, 0.0, motor.get_position(), 0.0, 0.0});
        for (const auto& motor : arm->get_gripper().get_motors())
            grip_cmds.push_back({0.0, 0.0, motor.get_position(), 0.0, 0.0});
        arm->get_arm().mit_control_all(arm_cmds);
        arm->get_gripper().mit_control_all(grip_cmds);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return arm;
}

std::vector<Motor> motor_snapshots(const Arm& left, const Arm& right) {
    // get_motors() deliberately returns a vector by value.  Never retain
    // pointers/references to its elements: they become dangling as soon as the
    // temporary vector is destroyed and can corrupt the measured start pose.
    std::vector<Motor> result;
    auto append = [&](const Arm& value) {
        // OpenArm accessors are not const in the upstream API.
        auto& mutable_value = const_cast<Arm&>(value);
        const auto arm_motors = mutable_value.get_arm().get_motors();
        const auto gripper_motors = mutable_value.get_gripper().get_motors();
        result.insert(result.end(), arm_motors.begin(), arm_motors.end());
        result.insert(result.end(), gripper_motors.begin(), gripper_motors.end());
    };
    append(left);
    append(right);
    return result;
}

std::array<double, kJointCount> measured_positions(Arm& left, Arm& right) {
    left.recv_all(200);
    right.recv_all(200);
    std::array<double, kJointCount> result{};
    const auto all = motor_snapshots(left, right);
    for (std::size_t i = 0; i < std::min(all.size(), result.size()); ++i)
        result[i] = all[i].get_position();
    return result;
}

void command_arm(Arm& arm, const double* q, const double* dq, bool damping_only = false) {
    std::vector<MIT> arm_commands, grip_commands;
    for (std::size_t i = 0; i < 7; ++i)
        arm_commands.push_back({damping_only ? 0.0 : kKp[i], kKd[i], q[i], dq[i], 0.0});
    grip_commands.push_back(
        {damping_only ? 0.0 : kKp[7], kKd[7], q[7], dq[7], 0.0});
    arm.get_arm().mit_control_all(arm_commands);
    arm.get_gripper().mit_control_all(grip_commands);
}

void damping_release(Arm& left, Arm& right) {
    const auto q = measured_positions(left, right);
    std::array<double, kJointCount> zero{};
    for (int cycle = 0; cycle < 250; ++cycle) {
        command_arm(left, q.data(), zero.data(), true);
        command_arm(right, q.data() + 8, zero.data() + 8, true);
        left.recv_all(200);
        right.recv_all(200);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    left.disable_all();
    right.disable_all();
}

void quintic_boundary_sample(double q0, double v0, double q1, double v1,
                             double s, double duration, double* q, double* dq) {
    s = std::clamp(s, 0.0, 1.0);
    const double s2 = s*s, s3 = s2*s, s4 = s3*s, s5 = s4*s;
    const double h00 = 1 - 10*s3 + 15*s4 - 6*s5;
    const double h01 = 10*s3 - 15*s4 + 6*s5;
    const double h10 = s - 6*s3 + 8*s4 - 3*s5;
    const double h11 = -4*s3 + 7*s4 - 3*s5;
    const double dh00 = -30*s2 + 60*s3 - 30*s4;
    const double dh01 = 30*s2 - 60*s3 + 30*s4;
    const double dh10 = 1 - 18*s2 + 32*s3 - 15*s4;
    const double dh11 = -12*s2 + 28*s3 - 15*s4;
    *q = q0*h00 + q1*h01 + duration*v0*h10 + duration*v1*h11;
    *dq = (q0*dh00 + q1*dh01) / duration + v0*dh10 + v1*dh11;
}

Frame interpolate(const Frame& a, const Frame& b, double time) {
    if (time <= a.time) return a;
    if (time >= b.time) return b;
    const double alpha = (time - a.time) / (b.time - a.time);
    Frame result;
    result.time = time;
    for (std::size_t i = 0; i < kJointCount; ++i) {
        result.q[i] = a.q[i] + alpha * (b.q[i] - a.q[i]);
        result.dq[i] = a.dq[i] + alpha * (b.dq[i] - a.dq[i]);
    }
    return result;
}

Frame sample_trajectory(const std::vector<Frame>& frames, std::size_t start,
                        double elapsed, std::size_t* cursor) {
    const double target = frames[start].time + elapsed;
    *cursor = std::max(*cursor, start);
    while (*cursor + 1 < frames.size() && frames[*cursor + 1].time < target) ++*cursor;
    if (*cursor + 1 >= frames.size()) return frames.back();
    return interpolate(frames[*cursor], frames[*cursor + 1], target);
}

std::string safety_reason(Arm& left, Arm& right,
                          const std::array<double, kJointCount>& command) {
    const auto all = motor_snapshots(left, right);
    if (all.size() != kJointCount) return "unexpected_motor_count";
    for (std::size_t i = 0; i < all.size(); ++i) {
        const auto& motor = all[i];
        // Damiao's upper status nibble also contains non-fault operating
        // states.  Only 8..14 are documented drive faults; treating every
        // non-zero value as a fault makes the normal state value 1 trip the
        // robot safety chain immediately after enabling.
        if (motor.get_error_code() >= 8 && motor.get_error_code() <= 14)
            return "motor_fault_q" + std::to_string(i) + "=" +
                   std::to_string(motor.get_error_code());
        if (motor.get_state_tmos() >= kMaxTemperature ||
            motor.get_state_trotor() >= kMaxTemperature)
            return "over_temperature_q" + std::to_string(i);
        const double measured_velocity_limit =
            (i == 7 || i == 15) ? 30.0 : kMaxMeasuredVelocity;
        if (std::abs(motor.get_velocity()) > measured_velocity_limit)
            return "measured_velocity_q" + std::to_string(i) + "=" +
                   std::to_string(motor.get_velocity()) + "_limit=" +
                   std::to_string(measured_velocity_limit);
        if (std::abs(motor.get_position() - command[i]) > kMaxTrackingError)
            return "tracking_error_q" + std::to_string(i);
    }
    return {};
}

void send_telemetry(const UdpSender& sender, double time, Arm& left, Arm& right) {
    const auto all = motor_snapshots(left, right);
    std::ostringstream out;
    out << time;
    for (int i = 0; i < 21; ++i) out << ",0";
    for (const auto& motor : all) out << ',' << motor.get_error_code();
    for (const auto& motor : all) out << ',' << motor.get_state_tmos();
    for (const auto& motor : all) out << ',' << motor.get_state_trotor();
    for (const auto& motor : all) out << ',' << motor.get_position();
    sender.send(out.str());
}

struct Options {
    std::string left_can = "can1";
    std::string right_can = "can0";
    uint16_t command_port = 47970;
    uint16_t status_port = 47971;
};

Options parse_options(int argc, char** argv) {
    Options result;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string key = argv[i];
        const std::string value = argv[i + 1];
        if (key == "--left-can") result.left_can = value;
        else if (key == "--right-can") result.right_can = value;
        else if (key == "--command-port") result.command_port = std::stoi(value);
        else if (key == "--status-port") result.status_port = std::stoi(value);
        else throw std::runtime_error("unknown option: " + key);
    }
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    try {
        const Options options = parse_options(argc, argv);
        UdpReceiver commands(options.command_port);
        UdpSender status(true);
        status.set_port(options.status_port);
        UdpSender telemetry;
        auto left = initialize_arm(options.left_can);
        auto right = initialize_arm(options.right_can);
        std::vector<Frame> trajectory;
        std::size_t prepared_frame = 0;
        bool prepared = false;
        bool executing = false;
        bool holding = false;
        bool transitioning = false;
        auto last_heartbeat = Clock::now();
        bool heartbeat_seen = false;
        auto phase_start = Clock::now();
        std::array<double, kJointCount> transition_start{};
        std::array<double, kJointCount> transition_start_dq{};
        std::size_t cursor = 0;
        double execution_time = 0.0;
        double transition_duration = kTransitionSeconds;
        double command_velocity_limit = kMaxCommandVelocity;
        double command_acceleration_limit = kMaxCommandAcceleration;
        std::array<double, kJointCount> held_q = measured_positions(*left, *right);
        // A connected service continuously holds the freshly measured pose;
        // it never commands a predefined home pose.
        holding = true;
        const auto period = std::chrono::microseconds(1000);
        auto next_tick = Clock::now();
        status.send("ready|independent_motion_studio_controller_v1");

        while (running) {
            for (const auto& message : commands.receive_all()) {
                const auto parts = split(message, '|');
                const std::string command = parts.empty() ? "" : parts[0];
                if (command == "heartbeat") {
                    last_heartbeat = Clock::now();
                    heartbeat_seen = true;
                } else if (command == "shutdown" || command == "stop") {
                    status.send("stopping|" + command);
                    running = false;
                } else if (command == "limits" && parts.size() >= 3 && !executing) {
                    try {
                        const double velocity = std::stod(parts[1]);
                        const double acceleration = std::stod(parts[2]);
                        if (!(velocity > 0.0 && velocity <= kHighDynamicsMaxVelocity &&
                              acceleration > 0.0 && acceleration <= kHighDynamicsMaxAcceleration))
                            throw std::runtime_error("limits_out_of_allowed_range");
                        command_velocity_limit = velocity;
                        command_acceleration_limit = acceleration;
                        status.send("limits|velocity=" + std::to_string(velocity) +
                                    "|acceleration=" + std::to_string(acceleration));
                    } catch (const std::exception& error) {
                        status.send(std::string("rejected|") + error.what());
                    }
                } else if (command == "load" && parts.size() >= 3 && !executing) {
                    try {
                        trajectory = load_csv(parts[1]);
                        // CSV validation can be slower than the watchdog window;
                        // resume supervision immediately after the atomic load.
                        last_heartbeat = Clock::now();
                        telemetry.set_port(static_cast<uint16_t>(std::stoi(parts[2])));
                        prepared = false;
                        status.send("loaded|frames=" + std::to_string(trajectory.size()));
                    } catch (const std::exception& error) {
                        trajectory.clear();
                        status.send(std::string("load_failed|") + error.what());
                    }
                } else if (command == "prepare" && parts.size() >= 2 && !executing) {
                    try {
                        const auto frame = static_cast<std::size_t>(std::stoull(parts[1]));
                        if (frame >= trajectory.size()) throw std::runtime_error("frame_out_of_range");
                        prepared_frame = frame;
                        prepared = true;
                        status.send("prepared|frame=" + std::to_string(frame) +
                                    "|time=" + std::to_string(trajectory[frame].time));
                    } catch (const std::exception& error) {
                        status.send(std::string("prepare_failed|") + error.what());
                    }
                } else if (command == "execute" && prepared && !executing) {
                    transition_start = measured_positions(*left, *right);
                    // The service is holding before execute, so the safe
                    // boundary condition is rest.  Raw single-frame velocity
                    // feedback can contain startup noise and make a quintic
                    // overshoot even when its positions are valid.
                    transition_start_dq.fill(0.0);
                    transition_duration = kTransitionSeconds;
                    for (std::size_t i = 0; i < kJointCount; ++i) {
                        if (!std::isfinite(transition_start[i])) {
                            transition_duration = 0.0;
                            break;
                        }
                        const double distance = std::abs(
                            trajectory[prepared_frame].q[i] - transition_start[i]);
                        // Rest-to-rest minimum-jerk extrema:
                        // vmax = 1.875 D/T, amax ~= 5.774 D/T^2.
                        transition_duration = std::max(
                            transition_duration,
                            1.05 * 1.875 * distance / command_velocity_limit);
                        transition_duration = std::max(
                            transition_duration,
                            1.05 * std::sqrt(
                                5.774 * distance / command_acceleration_limit));
                    }
                    std::string transition_error;
                    if (transition_duration <= 0.0)
                        transition_error = "invalid_measured_start_pose";
                    const int transition_steps = std::max(
                        1, static_cast<int>(std::ceil(transition_duration * kControlHz)));
                    for (int step = 0;
                         step <= transition_steps && transition_error.empty(); ++step) {
                        const double s = static_cast<double>(step) / transition_steps;
                        for (std::size_t i = 0; i < kJointCount; ++i) {
                            double checked_q = 0.0, checked_dq = 0.0;
                            quintic_boundary_sample(
                                transition_start[i], transition_start_dq[i],
                                trajectory[prepared_frame].q[i],
                                0.0, s,
                                transition_duration, &checked_q, &checked_dq);
                            const auto local = i % kJointsPerArm;
                            // A measured pose may already sit slightly outside
                            // the editor's conservative soft limit.  Permit a
                            // monotonic return from that exact pose to a valid
                            // target instead of making recovery impossible.
                            const double allowed_min = std::min(
                                kPositionMin[local], transition_start[i]);
                            const double allowed_max = std::max(
                                kPositionMax[local], transition_start[i]);
                            if (checked_q < allowed_min - 1e-6 ||
                                checked_q > allowed_max + 1e-6)
                                transition_error = "transition_position_limit_q" + std::to_string(i);
                            else if (std::abs(checked_dq) > command_velocity_limit)
                                transition_error = "transition_velocity_limit_q" + std::to_string(i);
                        }
                    }
                    if (!transition_error.empty()) {
                        status.send("transition_failed|" + transition_error);
                        prepared = false;
                        continue;
                    }
                    phase_start = Clock::now();
                    last_heartbeat = phase_start;
                    transitioning = true;
                    executing = true;
                    cursor = prepared_frame;
                    status.send("transition_started|frame=" + std::to_string(prepared_frame) +
                                "|duration=" + std::to_string(transition_duration));
                } else if (command != "heartbeat") {
                    status.send("rejected|state_or_command_invalid");
                }
            }

            const auto now = Clock::now();
            if (heartbeat_seen && now - last_heartbeat > kHeartbeatTimeout) {
                status.send("watchdog|heartbeat_timeout");
                running = false;
                break;
            }

            std::array<double, kJointCount> q{};
            std::array<double, kJointCount> dq{};
            if (executing) {
                if (transitioning) {
                    const double elapsed = std::chrono::duration<double>(now - phase_start).count();
                    const double s = std::clamp(elapsed / transition_duration, 0.0, 1.0);
                    for (std::size_t i = 0; i < kJointCount; ++i) {
                        quintic_boundary_sample(
                            transition_start[i], transition_start_dq[i],
                            trajectory[prepared_frame].q[i],
                            0.0, s,
                            transition_duration, &q[i], &dq[i]);
                    }
                    execution_time = elapsed;
                    if (s >= 1.0) {
                        transitioning = false;
                        phase_start = now;
                        status.send("started|frame=" + std::to_string(prepared_frame));
                    }
                } else {
                    const double elapsed = std::chrono::duration<double>(now - phase_start).count();
                    const Frame frame = sample_trajectory(trajectory, prepared_frame, elapsed, &cursor);
                    // Strict replay: the rendered editor trajectory is the
                    // command.  Do not phase-shift, chase, clamp, or otherwise
                    // rewrite q/dq online.  Runtime safety remains monitoring-
                    // only and releases the robot on a real violation.
                    q = frame.q;
                    dq = frame.dq;
                    execution_time = transition_duration + elapsed;
                    if (cursor + 1 >= trajectory.size() &&
                        frame.time >= trajectory.back().time) {
                        status.send("finished|frame=" + std::to_string(trajectory.size() - 1));
                        executing = false;
                        prepared = false;
                        holding = true;
                        held_q = q;
                        last_heartbeat = now;
                    }
                }
                command_arm(*left, q.data(), dq.data());
                command_arm(*right, q.data() + 8, dq.data() + 8);
                left->recv_all(200);
                right->recv_all(200);
                const std::string reason = safety_reason(*left, *right, q);
                if (!reason.empty()) {
                    status.send("safety|" + reason);
                    running = false;
                }
                send_telemetry(telemetry, execution_time, *left, *right);
            } else if (holding) {
                command_arm(*left, held_q.data(), dq.data());
                command_arm(*right, held_q.data() + 8, dq.data() + 8);
                left->recv_all(200);
                right->recv_all(200);
                const std::string reason = safety_reason(*left, *right, held_q);
                if (!reason.empty()) {
                    status.send("safety|" + reason);
                    running = false;
                }
                send_telemetry(telemetry, execution_time, *left, *right);
            } else {
                left->recv_all(200);
                right->recv_all(200);
            }
            next_tick += period;
            std::this_thread::sleep_until(next_tick);
            if (Clock::now() - next_tick > std::chrono::milliseconds(20)) next_tick = Clock::now();
        }
        status.send("release_started");
        damping_release(*left, *right);
        status.send("shutdown_done");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[motion-studio-hardware][FATAL] " << error.what() << std::endl;
        return 1;
    }
}
