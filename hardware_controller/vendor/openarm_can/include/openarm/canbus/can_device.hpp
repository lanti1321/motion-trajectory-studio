// Copyright 2025 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <linux/can.h>
#include <linux/can/raw.h>

#include <cstdint>
#include <vector>

namespace openarm::canbus {
// Abstract base class for CAN devices
class CANDevice {
public:
    explicit CANDevice(canid_t send_can_id, canid_t recv_can_id, canid_t recv_can_mask,
                       bool is_fd_enabled = false)
        : send_can_id_(send_can_id),
          recv_can_id_(recv_can_id),
          recv_can_mask_(recv_can_mask),
          is_fd_enabled_(is_fd_enabled) {}
    virtual ~CANDevice() = default;

    virtual void callback(const can_frame& frame) = 0;
    virtual void callback(const canfd_frame& frame) = 0;

    canid_t get_send_can_id() const { return send_can_id_; }
    canid_t get_recv_can_id() const { return recv_can_id_; }
    canid_t get_recv_can_mask() const { return recv_can_mask_; }
    bool is_fd_enabled() const { return is_fd_enabled_; }

protected:
    canid_t send_can_id_;
    canid_t recv_can_id_;
    // mask for receiving
    canid_t recv_can_mask_ = CAN_SFF_MASK;
    bool is_fd_enabled_ = false;
};
}  // namespace openarm::canbus
