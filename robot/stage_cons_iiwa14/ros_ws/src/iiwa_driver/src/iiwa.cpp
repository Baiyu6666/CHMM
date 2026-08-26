//|
//|    Copyright (C) 2019 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
//|    Authors:  Konstantinos Chatzilygeroudis (maintainer)
//|              Bernardo Fichera
//|              Walid Amanhoud
//|    email:    costashatz@gmail.com
//|              bernardo.fichera@epfl.ch
//|              walid.amanhoud@epfl.ch
//|    Other contributors:
//|              Yoan Mollard (yoan@aubrune.eu)
//|    website:  lasa.epfl.ch
//|
//|    This file is part of iiwa_ros.
//|
//|    iiwa_ros is free software: you can redistribute it and/or modify
//|    it under the terms of the GNU General Public License as published by
//|    the Free Software Foundation, either version 3 of the License, or
//|    (at your option) any later version.
//|
//|    iiwa_ros is distributed in the hope that it will be useful,
//|    but WITHOUT ANY WARRANTY; without even the implied warranty of
//|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//|    GNU General Public License for more details.
//|
#include <iiwa_driver/iiwa.h>

// ROS Headers
#include <control_toolbox/filters.h>
#include <controller_manager/controller_manager.h>

#include <urdf/model.h>

// FRI Headers
#include <kuka/fri/ClientData.h>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <unistd.h>

namespace {
    const char* fri_session_state_name(kuka::fri::ESessionState state)
    {
        switch (state) {
        case kuka::fri::IDLE: return "IDLE";
        case kuka::fri::MONITORING_WAIT: return "MONITORING_WAIT";
        case kuka::fri::MONITORING_READY: return "MONITORING_READY";
        case kuka::fri::COMMANDING_WAIT: return "COMMANDING_WAIT";
        case kuka::fri::COMMANDING_ACTIVE: return "COMMANDING_ACTIVE";
        default: return "UNKNOWN";
        }
    }

    const char* fri_connection_quality_name(kuka::fri::EConnectionQuality quality)
    {
        switch (quality) {
        case kuka::fri::POOR: return "POOR";
        case kuka::fri::FAIR: return "FAIR";
        case kuka::fri::GOOD: return "GOOD";
        case kuka::fri::EXCELLENT: return "EXCELLENT";
        default: return "UNKNOWN";
        }
    }
}

namespace iiwa_ros {
    Iiwa::Iiwa(ros::NodeHandle& nh)
    {
        init(nh);
    }

    Iiwa::~Iiwa()
    {
        // Disconnect from robot
        _disconnect_fri();

        // Delete FRI message data
        if (_fri_message_data)
            delete _fri_message_data;
    }

    void Iiwa::init(ros::NodeHandle& nh)
    {
        _fri_message_data = nullptr;
        _initialized = false;
        _idle = true;
        _commanding = false;
        _client_command_mode = 0;
        _demo_mode_requested.store(false);
        _last_demo_heartbeat_wall_sec.store(0.0);
        _demo_mode_active = false;
        _position_command_enabled.store(false);
        _position_arm_requested.store(false);
        _position_controller_reset_requested.store(0);
        _position_controller_reset_completed.store(0);
        _position_hold_valid.store(false);
        _last_position_heartbeat_wall_sec.store(0.0);
        _control_cycles = 0;
        _cycle_period_deadline_misses = 0;
        _control_work_overruns = 0;
        _last_cycle_start_wall_sec = 0.0;
        _last_cycle_period_sec = 0.0;
        _maximum_cycle_period_sec = 0.0;
        _last_control_work_sec = 0.0;
        _maximum_control_work_sec = 0.0;
        _last_deadline_miss_wall_sec = 0.0;
        _last_fri_diagnostics_publish_wall_sec = 0.0;
        _last_status_publish_wall_sec = 0.0;
        _fri_realtime_enabled = false;
        _fri_realtime_effective_priority = 0;
        _fri_effective_cpu_affinity = -1;
        _connection_closed_failures = 0;
        _receive_failures = 0;
        _decode_failures = 0;
        _message_id_failures = 0;
        _encode_failures = 0;
        _send_failures = 0;
        _monitor_sequence_gaps = 0;
        _duplicate_monitor_messages = 0;
        _monitor_sequence_resets = 0;
        _last_monitor_sequence = 0;
        _have_monitor_sequence = false;
        _last_io_failure_wall_sec = 0.0;
        _fri_session_state = static_cast<int>(kuka::fri::IDLE);
        _fri_connection_quality = static_cast<int>(kuka::fri::POOR);
        _fri_safety_state = 0;
        _fri_operation_mode = 0;
        _fri_drive_state = 0;
        _fri_sample_time_sec = 0.0;
        _fri_receive_multiplier = 0;
        _fri_session_state_changes = 0;
        _last_fri_session_state_change_wall_sec = 0.0;
        _nh = nh;
        _ns = nh.getNamespace();
        _load_params(); // load parameters
        _init(); // initialize
        // add namespace
        _commanding_status_pub.init(_nh, _ns+"/commanding_status", 4, true);
        _demo_mode_status_pub.init(_nh, _ns+"/demo_mode_active", 4, true);
        _client_command_mode_pub.init(_nh, _ns+"/fri_command_mode", 4, true);
        _fri_diagnostics_pub.init(_nh, _ns+"/fri_diagnostics", 4);
        _demo_mode_service = _nh.advertiseService(
            "iiwa_driver/set_demo_mode", &Iiwa::_set_demo_mode, this);
        _position_command_service = _nh.advertiseService(
            "iiwa_driver/set_position_commanding", &Iiwa::_set_position_commanding, this);
        _position_heartbeat_sub = _nh.subscribe<std_msgs::Empty>(
            "iiwa_driver/position_command_heartbeat", 1,
            &Iiwa::_position_command_heartbeat, this);
        _demo_heartbeat_sub = _nh.subscribe<std_msgs::Empty>(
            "iiwa_driver/demo_mode_heartbeat", 1, &Iiwa::_demo_heartbeat, this);
        _controller_manager.reset(new controller_manager::ControllerManager(this, _nh));

        if (_init_fri())
            _initialized = true;
        else
            _initialized = false;
    }

    void Iiwa::run()
    {
        if (!_initialized) {
            ROS_ERROR_STREAM("Not connected to the robot. Cannot run!");
            return;
        }

        std::thread t1(&Iiwa::_ctrl_loop, this);
        t1.join();
    }

    bool Iiwa::initialized()
    {
        return _initialized;
    }

    void Iiwa::_init()
    {
        // Get joint names
        _num_joints = _joint_names.size();

        // Resize vectors
        _joint_position.resize(_num_joints);
        _joint_velocity.resize(_num_joints);
        _joint_effort.resize(_num_joints);
        _joint_position_command.resize(_num_joints);
        _position_hold_command.resize(_num_joints);
        _joint_velocity_command.resize(_num_joints);
        _joint_effort_command.resize(_num_joints);

        // Get the URDF XML from the parameter server
        urdf::Model urdf_model;
        std::string urdf_string;

        // search and wait for robot_description on param server
        while (urdf_string.empty()) {
            ROS_INFO_ONCE_NAMED("Iiwa", "Iiwa is waiting for model"
                                        " URDF in parameter [%s] on the ROS param server.",
                _robot_description.c_str());

            if(_nh.getParam(_robot_description, urdf_string)){
                ROS_INFO_STREAM("Got parameter: " + _robot_description);
            }
            else{
                ROS_ERROR_STREAM("Parameter " + _robot_description + " not found");
            }

            usleep(100000);
        }
        ROS_INFO_STREAM_NAMED("Iiwa", "Received urdf from param server, parsing...");

        const urdf::Model* const urdf_model_ptr = urdf_model.initString(urdf_string) ? &urdf_model : nullptr;
        if (urdf_model_ptr == nullptr)
            ROS_WARN_STREAM_NAMED("Iiwa", "Could not read URDF from '" << _robot_description << "' parameters. Joint limits will not work.");

        // Initialize Controller
        for (int i = 0; i < _num_joints; ++i) {
            _joint_position[i] = _joint_velocity[i] = _joint_effort[i] = 0.;
            _joint_position_command[i] = _joint_velocity_command[i]
                = _joint_effort_command[i] = 0.;
            // Create joint state interface
            hardware_interface::JointStateHandle joint_state_handle(_joint_names[i], &_joint_position[i], &_joint_velocity[i], &_joint_effort[i]);
            _joint_state_interface.registerHandle(joint_state_handle);

            // Get joint limits from URDF
            bool has_soft_limits = false;
            bool has_limits = urdf_model_ptr != nullptr;
            joint_limits_interface::JointLimits limits;
            joint_limits_interface::SoftJointLimits soft_limits;

            if (has_limits) {
                auto urdf_joint = urdf_model_ptr->getJoint(_joint_names[i]);
                if (!urdf_joint) {
                    ROS_WARN_STREAM_NAMED("Iiwa", "Could not find joint '" << _joint_names[i] << "' in URDF. No limits will be applied for this joint.");
                    continue;
                }

                getJointLimits(urdf_joint, limits);
                if (getSoftJointLimits(urdf_joint, soft_limits))
                    has_soft_limits = true;
            }

            // Create position joint interface
            hardware_interface::JointHandle joint_position_handle(joint_state_handle, &_joint_position_command[i]);

            if (has_soft_limits) {
                joint_limits_interface::PositionJointSoftLimitsHandle joint_limits_handle(joint_position_handle, limits, soft_limits);
                _position_joint_limits_interface.registerHandle(joint_limits_handle);
            }
            else {
                joint_limits_interface::PositionJointSaturationHandle joint_limits_handle(joint_position_handle, limits);
                _position_joint_saturation_interface.registerHandle(joint_limits_handle);
            }

            _position_joint_interface.registerHandle(joint_position_handle);

            // Create effort joint interface
            hardware_interface::JointHandle joint_effort_handle(joint_state_handle, &_joint_effort_command[i]);

            if (has_soft_limits) {
                joint_limits_interface::EffortJointSoftLimitsHandle joint_limits_handle(joint_effort_handle, limits, soft_limits);
                _effort_joint_limits_interface.registerHandle(joint_limits_handle);
            }
            else if (has_limits) {
                joint_limits_interface::EffortJointSaturationHandle joint_limits_handle(joint_effort_handle, limits);
                _effort_joint_saturation_interface.registerHandle(joint_limits_handle);
            }

            _effort_joint_interface.registerHandle(joint_effort_handle);

            // Create velocity joint interface
            hardware_interface::JointHandle joint_velocity_handle(joint_state_handle, &_joint_velocity_command[i]);

            if (has_soft_limits) {
                joint_limits_interface::VelocityJointSoftLimitsHandle joint_limits_handle(joint_velocity_handle, limits, soft_limits);
                _velocity_joint_limits_interface.registerHandle(joint_limits_handle);
            }
            else {
                joint_limits_interface::VelocityJointSaturationHandle joint_limits_handle(joint_velocity_handle, limits);
                _velocity_joint_saturation_interface.registerHandle(joint_limits_handle);
            }

            _velocity_joint_interface.registerHandle(joint_velocity_handle);
        }

        registerInterface(&_joint_state_interface);
        registerInterface(&_position_joint_interface);
        registerInterface(&_effort_joint_interface);
        registerInterface(&_velocity_joint_interface);

        _additional_pub.init(_nh, "additional_outputs", 20);
        _additional_pub.msg_.external_torques.layout.dim.resize(1);
        _additional_pub.msg_.external_torques.layout.data_offset = 0;
        _additional_pub.msg_.external_torques.layout.dim[0].size = _num_joints;
        _additional_pub.msg_.external_torques.layout.dim[0].stride = 0;
        _additional_pub.msg_.external_torques.data.resize(_num_joints);
        _additional_pub.msg_.commanded_torques.layout.dim.resize(1);
        _additional_pub.msg_.commanded_torques.layout.data_offset = 0;
        _additional_pub.msg_.commanded_torques.layout.dim[0].size = _num_joints;
        _additional_pub.msg_.commanded_torques.layout.dim[0].stride = 0;
        _additional_pub.msg_.commanded_torques.data.resize(_num_joints);
        _additional_pub.msg_.commanded_positions.layout.dim.resize(1);
        _additional_pub.msg_.commanded_positions.layout.data_offset = 0;
        _additional_pub.msg_.commanded_positions.layout.dim[0].size = _num_joints;
        _additional_pub.msg_.commanded_positions.layout.dim[0].stride = 0;
        _additional_pub.msg_.commanded_positions.data.resize(_num_joints);
    }

    void Iiwa::_ctrl_loop()
    {
        if (!_configure_control_thread()) {
            ROS_FATAL(
                "FRI control thread realtime setup failed; refusing to run the robot driver.");
            ros::shutdown();
            return;
        }
        static ros::Rate rate(_control_freq);
        const double expected_period = 1.0 / _control_freq;
        const double deadline_threshold = _fri_deadline_miss_factor * expected_period;
        const double minimum_elapsed = 0.25 * expected_period;
        const double maximum_elapsed = 4.0 * expected_period;
        std::chrono::steady_clock::time_point previous_cycle_start;
        bool have_previous_cycle_start = false;
        while (ros::ok()) {
            const std::chrono::steady_clock::time_point cycle_start_steady =
                std::chrono::steady_clock::now();
            const double cycle_start_wall_sec = ros::WallTime::now().toSec();
            if (_last_cycle_start_wall_sec > 0.0) {
                _last_cycle_period_sec = cycle_start_wall_sec - _last_cycle_start_wall_sec;
                _maximum_cycle_period_sec = std::max(
                    _maximum_cycle_period_sec, _last_cycle_period_sec);
                if (_last_cycle_period_sec > deadline_threshold) {
                    ++_cycle_period_deadline_misses;
                    _last_deadline_miss_wall_sec = cycle_start_wall_sec;
                }
            }
            _last_cycle_start_wall_sec = cycle_start_wall_sec;
            ++_control_cycles;
            ros::Time time = ros::Time::now();

            double measured_elapsed = expected_period;
            if (have_previous_cycle_start) {
                measured_elapsed = std::chrono::duration<double>(
                    cycle_start_steady - previous_cycle_start).count();
            }
            previous_cycle_start = cycle_start_steady;
            have_previous_cycle_start = true;
            if (!std::isfinite(measured_elapsed) || measured_elapsed <= 0.0) {
                measured_elapsed = expected_period;
            }
            // Feed controllers and joint-limit interfaces the actual monotonic
            // cycle duration. Bound a one-off network stall so it cannot cause
            // one oversized rate-limit step after reconnection.
            const double bounded_elapsed = std::max(
                minimum_elapsed, std::min(measured_elapsed, maximum_elapsed));
            const ros::Duration elapsed_time(bounded_elapsed);

            const bool was_commanding = _commanding.load();
            _read(elapsed_time);
            const bool commanding_started = !was_commanding && _commanding.load();
            const std::uint64_t reset_requested =
                _position_controller_reset_requested.load();
            const bool position_reset_requested =
                reset_requested != _position_controller_reset_completed.load();
            _update_demo_motion_gate();
            // Controllers are launched before the SmartPAD FRI session and may
            // therefore have initialized their command buffers from zero/stale
            // joints.  Reset them exactly when the first COMMANDING_ACTIVE
            // sample provides real measured joints.  The position-command gate
            // is still closed, so this cannot enable robot motion by itself.
            _controller_manager->update(
                ros::Time::now(), elapsed_time,
                commanding_started || position_reset_requested);
            if (commanding_started || position_reset_requested) {
                // Capture on the control thread, where the measured vector is
                // owned.  A closed POSITION gate then holds this one snapshot
                // instead of chasing the changing measured position.
                for (int i = 0; i < _num_joints; ++i) {
                    _position_hold_command[i] = _joint_position[i];
                }
                _position_hold_valid.store(true);
            }
            if (position_reset_requested) {
                _position_controller_reset_completed.store(reset_requested);
                ROS_INFO(
                    "Reset position controller and latched a fixed measured-joint hold.");
            }
            if (commanding_started) {
                ROS_INFO("Reset active controllers from the first COMMANDING_ACTIVE joint sample.");
            }
            _write(elapsed_time);

            // publish additional outputs
            if (_additional_pub.trylock()) {
                _additional_pub.msg_.header.stamp = ros::Time::now();
                for (unsigned i = 0; i < _num_joints; i++) {
                    _additional_pub.msg_.external_torques.data[i] = _robot_state.getExternalTorque()[i];
                    _additional_pub.msg_.commanded_torques.data[i] = _robot_state.getCommandedTorque()[i];
                    _additional_pub.msg_.commanded_positions.data[i] = _robot_state.getCommandedJointPosition()[i];
                }
                _additional_pub.unlockAndPublish();
            }

            _publish();
            _last_control_work_sec = ros::WallTime::now().toSec() - cycle_start_wall_sec;
            _maximum_control_work_sec = std::max(
                _maximum_control_work_sec, _last_control_work_sec);
            if (_last_control_work_sec > expected_period) {
                ++_control_work_overruns;
                _last_deadline_miss_wall_sec = ros::WallTime::now().toSec();
            }
            _publish_fri_diagnostics();
            rate.sleep();
        }
    }

    bool Iiwa::_configure_control_thread()
    {
        const pthread_t thread = pthread_self();
        const int name_result = pthread_setname_np(thread, "iiwa_fri_ctrl");
        if (name_result != 0) {
            ROS_ERROR_STREAM("Could not name the FRI control thread: "
                             << std::strerror(name_result));
            return false;
        }

        const long online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (_fri_cpu_affinity < 0 || _fri_cpu_affinity >= online_cpus
            || _fri_cpu_affinity >= CPU_SETSIZE) {
            ROS_ERROR_STREAM("Invalid FRI CPU affinity " << _fri_cpu_affinity
                             << "; online CPU count is " << online_cpus << ".");
            return false;
        }
        cpu_set_t requested_cpus;
        CPU_ZERO(&requested_cpus);
        CPU_SET(_fri_cpu_affinity, &requested_cpus);
        const int affinity_result = pthread_setaffinity_np(
            thread, sizeof(requested_cpus), &requested_cpus);
        if (affinity_result != 0) {
            ROS_ERROR_STREAM("Could not bind the FRI control thread to CPU "
                             << _fri_cpu_affinity << ": "
                             << std::strerror(affinity_result));
            return false;
        }

        const int minimum_priority = sched_get_priority_min(SCHED_FIFO);
        const int maximum_priority = sched_get_priority_max(SCHED_FIFO);
        if (_fri_realtime_priority < minimum_priority
            || _fri_realtime_priority > maximum_priority) {
            ROS_ERROR_STREAM("Invalid FRI SCHED_FIFO priority "
                             << _fri_realtime_priority << "; valid range is "
                             << minimum_priority << "-" << maximum_priority << ".");
            return false;
        }
        sched_param requested_schedule = {};
        requested_schedule.sched_priority = _fri_realtime_priority;
        const int schedule_result = pthread_setschedparam(
            thread, SCHED_FIFO, &requested_schedule);
        if (schedule_result != 0) {
            ROS_ERROR_STREAM("Could not enable SCHED_FIFO for the FRI control thread: "
                             << std::strerror(schedule_result)
                             << ". Verify cap_sys_nice on the iiwa_driver binary.");
            return false;
        }

        int effective_policy = SCHED_OTHER;
        sched_param effective_schedule = {};
        const int get_schedule_result = pthread_getschedparam(
            thread, &effective_policy, &effective_schedule);
        cpu_set_t effective_cpus;
        CPU_ZERO(&effective_cpus);
        const int get_affinity_result = pthread_getaffinity_np(
            thread, sizeof(effective_cpus), &effective_cpus);
        _fri_realtime_enabled = get_schedule_result == 0
            && effective_policy == SCHED_FIFO
            && effective_schedule.sched_priority == _fri_realtime_priority;
        _fri_realtime_effective_priority = get_schedule_result == 0
            ? effective_schedule.sched_priority : 0;
        _fri_effective_cpu_affinity = get_affinity_result == 0
            && CPU_ISSET(_fri_cpu_affinity, &effective_cpus)
            && CPU_COUNT(&effective_cpus) == 1
            ? _fri_cpu_affinity : -1;
        if (!_fri_realtime_enabled
            || _fri_effective_cpu_affinity != _fri_cpu_affinity) {
            ROS_ERROR_STREAM("FRI control thread realtime configuration did not verify: "
                             "policy=" << effective_policy
                             << ", priority=" << _fri_realtime_effective_priority
                             << ", CPU=" << _fri_effective_cpu_affinity << ".");
            return false;
        }

        ROS_INFO_STREAM("FRI control thread enabled with SCHED_FIFO priority "
                        << _fri_realtime_effective_priority << " on CPU "
                        << _fri_effective_cpu_affinity << ".");
        return true;
    }

    void Iiwa::_publish()
    {
        const double now = ros::WallTime::now().toSec();
        if (now - _last_status_publish_wall_sec < _status_publish_period) {
            return;
        }
        _last_status_publish_wall_sec = now;
        if (_commanding_status_pub.trylock()) {
            _commanding_status_pub.msg_.data = _commanding.load();
            _commanding_status_pub.unlockAndPublish();
        }
        if (_demo_mode_status_pub.trylock()) {
            _demo_mode_status_pub.msg_.data = _demo_mode_active.load();
            _demo_mode_status_pub.unlockAndPublish();
        }
        if (_client_command_mode_pub.trylock()) {
            _client_command_mode_pub.msg_.data = _client_command_mode.load();
            _client_command_mode_pub.unlockAndPublish();
        }
    }

    void Iiwa::_publish_fri_diagnostics()
    {
        const double now = ros::WallTime::now().toSec();
        if (now - _last_fri_diagnostics_publish_wall_sec
            < _fri_diagnostics_publish_period) {
            return;
        }
        if (!_fri_diagnostics_pub.trylock()) {
            return;
        }
        _last_fri_diagnostics_publish_wall_sec = now;
        iiwa_driver::FriDiagnostics& message = _fri_diagnostics_pub.msg_;
        message.header.stamp = ros::Time::now();
        message.session_state = static_cast<std::uint8_t>(_fri_session_state);
        message.connection_quality = static_cast<std::uint8_t>(_fri_connection_quality);
        message.safety_state = static_cast<std::uint8_t>(_fri_safety_state);
        message.operation_mode = static_cast<std::uint8_t>(_fri_operation_mode);
        message.drive_state = static_cast<std::uint8_t>(_fri_drive_state);
        message.client_command_mode = _client_command_mode.load();
        message.commanding = _commanding.load();
        message.sample_time_sec = _fri_sample_time_sec;
        message.receive_multiplier = _fri_receive_multiplier;
        message.realtime_scheduling_enabled = _fri_realtime_enabled;
        message.realtime_priority = _fri_realtime_effective_priority;
        message.cpu_affinity = _fri_effective_cpu_affinity;
        message.control_cycles = _control_cycles;
        message.cycle_period_deadline_misses = _cycle_period_deadline_misses;
        message.control_work_overruns = _control_work_overruns;
        message.deadline_threshold_sec = _fri_deadline_miss_factor / _control_freq;
        message.last_cycle_period_sec = _last_cycle_period_sec;
        message.maximum_cycle_period_sec = _maximum_cycle_period_sec;
        message.last_control_work_sec = _last_control_work_sec;
        message.maximum_control_work_sec = _maximum_control_work_sec;
        message.last_deadline_miss_wall_sec = _last_deadline_miss_wall_sec;
        message.connection_closed_failures = _connection_closed_failures;
        message.receive_failures = _receive_failures;
        message.decode_failures = _decode_failures;
        message.message_id_failures = _message_id_failures;
        message.encode_failures = _encode_failures;
        message.send_failures = _send_failures;
        message.monitor_sequence_gaps = _monitor_sequence_gaps;
        message.duplicate_monitor_messages = _duplicate_monitor_messages;
        message.monitor_sequence_resets = _monitor_sequence_resets;
        message.last_monitor_sequence = _last_monitor_sequence;
        message.last_io_failure_wall_sec = _last_io_failure_wall_sec;
        message.session_state_changes = _fri_session_state_changes;
        message.last_session_state_change_wall_sec =
            _last_fri_session_state_change_wall_sec;
        _fri_diagnostics_pub.unlockAndPublish();
    }

    void Iiwa::_on_fri_state_change(
        kuka::fri::ESessionState old_state, kuka::fri::ESessionState current_state)
    {
        ++_fri_session_state_changes;
        _last_fri_session_state_change_wall_sec = ros::WallTime::now().toSec();
        const kuka::fri::EConnectionQuality quality =
            static_cast<kuka::fri::EConnectionQuality>(_fri_connection_quality);
        if (old_state == kuka::fri::COMMANDING_ACTIVE
            && current_state != kuka::fri::COMMANDING_ACTIVE) {
            ROS_ERROR_STREAM("FRI left COMMANDING_ACTIVE: "
                << fri_session_state_name(old_state) << " -> "
                << fri_session_state_name(current_state)
                << ", connection_quality=" << fri_connection_quality_name(quality)
                << ", monitor_sequence=" << _last_monitor_sequence
                << ", sequence_gaps=" << _monitor_sequence_gaps
                << ", receive_failures=" << _receive_failures
                << ", send_failures=" << _send_failures
                << ", deadline_misses=" << _cycle_period_deadline_misses
                << ", last_cycle_period_ms=" << 1000.0 * _last_cycle_period_sec
                << ", max_cycle_period_ms=" << 1000.0 * _maximum_cycle_period_sec
                << ".");
        }
        else {
            ROS_INFO_STREAM("FRI session state changed: "
                << fri_session_state_name(old_state) << " -> "
                << fri_session_state_name(current_state)
                << ", connection_quality=" << fri_connection_quality_name(quality)
                << ", monitor_sequence=" << _last_monitor_sequence << ".");
        }
    }

    bool Iiwa::_set_demo_mode(std_srvs::SetBool::Request& request,
        std_srvs::SetBool::Response& response)
    {
        if (request.data) {
            if (_position_arm_requested.load() || _position_command_enabled.load()) {
                response.success = false;
                response.message =
                    "Demo/Torque mode rejected while Position command ownership is armed.";
                return true;
            }
            if (!_commanding.load()
                || _client_command_mode.load() != static_cast<int>(kuka::fri::TORQUE)) {
                response.success = false;
                response.message =
                    "Start FRIOverlayGripper in COMMANDING_ACTIVE + TORQUE before enabling Demo motion.";
                return true;
            }
        }
        _last_demo_heartbeat_wall_sec.store(ros::WallTime::now().toSec());
        _demo_mode_requested.store(request.data);
        response.success = true;
        response.message = request.data
            ? "Demo acquisition mode requested; mechanical position hold does not exist."
            : "Demo acquisition mode disabled; mechanical position hold does not exist.";
        return true;
    }

    bool Iiwa::_set_position_commanding(std_srvs::SetBool::Request& request,
        std_srvs::SetBool::Response& response)
    {
        if (!request.data) {
            _position_arm_requested.store(false);
            _position_hold_valid.store(false);
            _position_command_enabled.store(false);
            const std::uint64_t reset_sequence =
                _position_controller_reset_requested.fetch_add(1) + 1;
            const ros::WallTime deadline =
                ros::WallTime::now() + ros::WallDuration(0.25);
            while (
                ros::WallTime::now() < deadline
                && _position_controller_reset_completed.load() < reset_sequence
            ) {
                ros::WallDuration(0.005).sleep();
            }
            response.success =
                _position_controller_reset_completed.load() >= reset_sequence
                && _position_hold_valid.load();
            response.message = response.success
                ? "FRI position commands disabled; fixed hold latched at the measured joints."
                : "Timed out latching a fixed hold while the position command gate was closed.";
            return true;
        }
        if (_demo_mode_requested.load() || _demo_mode_active.load()) {
            response.success = false;
            response.message =
                "Position command ownership rejected while Demo/Torque motion is active.";
            return true;
        }
        _position_arm_requested.store(true);
        _last_position_heartbeat_wall_sec.store(ros::WallTime::now().toSec());
        const ros::WallTime deadline = ros::WallTime::now() + ros::WallDuration(0.25);
        while (ros::WallTime::now() < deadline && !_position_command_enabled.load()) {
            ros::WallDuration(0.005).sleep();
        }
        response.success = _position_command_enabled.load();
        response.message = response.success
            ? "FRI position commands armed."
            : "FRI position arm request was rejected or timed out.";
        if (!response.success)
            _position_arm_requested.store(false);
        return true;
    }

    void Iiwa::_position_command_heartbeat(const std_msgs::Empty::ConstPtr&)
    {
        _last_position_heartbeat_wall_sec.store(ros::WallTime::now().toSec());
    }

    void Iiwa::_demo_heartbeat(const std_msgs::Empty::ConstPtr&)
    {
        if (_demo_mode_requested.load()) {
            _last_demo_heartbeat_wall_sec.store(ros::WallTime::now().toSec());
        }
    }

    void Iiwa::_update_demo_motion_gate()
    {
        if (_idle) {
            return;
        }

        bool requested = _demo_mode_requested.load();
        if (requested) {
            const double heartbeat_age = ros::WallTime::now().toSec()
                - _last_demo_heartbeat_wall_sec.load();
            if (heartbeat_age > _demo_heartbeat_timeout) {
                ROS_ERROR_STREAM_THROTTLE(1.0,
                    "Demo heartbeat expired after " << heartbeat_age
                    << " s; disabling logical Demo acquisition mode.");
                _demo_mode_requested.store(false);
                requested = false;
            }
        }

        if (requested == _demo_mode_active.load()) {
            return;
        }
        _demo_mode_active.store(requested);
        ROS_INFO_STREAM("Logical Demo acquisition mode "
                        << (_demo_mode_active.load() ? "enabled" : "disabled")
                        << "; FRI position reference remains measured joints.");
    }

    void Iiwa::_load_params()
    {
        ros::NodeHandle n_p("~");

        n_p.param(_ns + "/iiwa_driver/fri/port", _port, 30200); // Default port is 30200
        n_p.param<std::string>(_ns + "/iiwa_driver/fri/robot_ip", _remote_host, "192.170.10.2"); // Default robot ip is 192.170.10.2
        n_p.param<std::string>(_ns + "/iiwa_driver/fri/robot_description", _robot_description, _ns + "/robot_description");

        n_p.param(_ns + "/iiwa_driver/hardware_interface/control_freq", _control_freq, 200.);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/status_publish_period",
            _status_publish_period, 0.05);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/fri_realtime_priority",
            _fri_realtime_priority, 80);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/fri_cpu_affinity",
            _fri_cpu_affinity, 4);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/fri_diagnostics_publish_period",
            _fri_diagnostics_publish_period, 1.0);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/fri_deadline_miss_factor",
            _fri_deadline_miss_factor, 1.5);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/demo_heartbeat_timeout",
            _demo_heartbeat_timeout, 0.5);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/position_arm_tolerance",
            _position_arm_tolerance, 0.00872664626);
        n_p.param(_ns + "/iiwa_driver/hardware_interface/position_heartbeat_timeout",
            _position_heartbeat_timeout, 0.2);
        if (!std::isfinite(_demo_heartbeat_timeout) || _demo_heartbeat_timeout <= 0.0) {
            ROS_WARN("Invalid demo heartbeat timeout; using 0.5 s.");
            _demo_heartbeat_timeout = 0.5;
        }
        if (!std::isfinite(_fri_diagnostics_publish_period)
            || _fri_diagnostics_publish_period <= 0.0) {
            ROS_WARN("Invalid FRI diagnostics publish period; using 1.0 s.");
            _fri_diagnostics_publish_period = 1.0;
        }
        if (!std::isfinite(_status_publish_period) || _status_publish_period <= 0.0) {
            ROS_WARN("Invalid driver status publish period; using 0.05 s.");
            _status_publish_period = 0.05;
        }
        if (!std::isfinite(_fri_deadline_miss_factor)
            || _fri_deadline_miss_factor < 1.0) {
            ROS_WARN("Invalid FRI deadline miss factor; using 1.5.");
            _fri_deadline_miss_factor = 1.5;
        }
        if (!std::isfinite(_position_arm_tolerance) || _position_arm_tolerance <= 0.0) {
            ROS_WARN("Invalid position arm tolerance; using 0.5 degree.");
            _position_arm_tolerance = 0.00872664626;
        }
        if (!std::isfinite(_position_heartbeat_timeout)
            || _position_heartbeat_timeout <= 0.0) {
            ROS_WARN("Invalid position heartbeat timeout; using 0.2 s.");
            _position_heartbeat_timeout = 0.2;
        }

        if(n_p.getParam(_ns + "/iiwa_driver/hardware_interface/joints", _joint_names)){
            ROS_INFO_STREAM_ONCE_NAMED("Iiwa","Got parameter hardware_interface/joints");
        }
        else{
            ROS_ERROR_STREAM_ONCE_NAMED("Iiwa","Parameter hardware_interface/joints not found");
        }
    }

    void Iiwa::_read(ros::Duration elapsed_time)
    {
        // Read data from robot (via FRI)
        const bool was_commanding = _commanding.load();
        kuka::fri::ESessionState fri_state = kuka::fri::IDLE;
        if (!_read_fri(fri_state)) {
            _idle = true;
            _commanding = false;
            _client_command_mode = 0;
            _demo_mode_requested.store(false);
            _demo_mode_active.store(false);
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            _position_hold_valid.store(false);
            return;
        }
        const int previous_command_mode = _client_command_mode.load();
        const int current_command_mode =
            static_cast<int>(_robot_state.getClientCommandMode());
        _client_command_mode.store(current_command_mode);
        if (current_command_mode != previous_command_mode) {
            // A SmartPAD mode transition creates a new control epoch. Neither a
            // Demo heartbeat nor a Position target from the preceding epoch may
            // regain ownership automatically after that transition.
            _demo_mode_requested.store(false);
            _demo_mode_active.store(false);
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            _position_hold_valid.store(false);
            ROS_WARN_STREAM("FRI command mode changed from " << previous_command_mode
                            << " to " << current_command_mode
                            << "; all robot command ownership was reset.");
        }
        if (current_command_mode != static_cast<int>(kuka::fri::POSITION)) {
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            _position_hold_valid.store(false);
        }
        if (current_command_mode != static_cast<int>(kuka::fri::TORQUE)) {
            _demo_mode_requested.store(false);
            _demo_mode_active.store(false);
        }

        switch (fri_state) {
        case kuka::fri::MONITORING_WAIT:
        case kuka::fri::MONITORING_READY:
        case kuka::fri::COMMANDING_WAIT:
            _idle = false;
            _commanding = false;
            _demo_mode_requested.store(false);
            _demo_mode_active.store(false);
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            break;
        case kuka::fri::COMMANDING_ACTIVE:
            _idle = false;
            _commanding = true;
            break;
        case kuka::fri::IDLE: // if idle, do nothing
        default:
            _idle = true;
            _commanding = false;
            _demo_mode_requested.store(false);
            _demo_mode_active.store(false);
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            return;
        }

        // Update ROS structures
        _joint_position_prev = _joint_position;

        for (int i = 0; i < _num_joints; i++) {
            _joint_position[i] = _robot_state.getMeasuredJointPosition()[i];
            _joint_velocity[i] = filters::exponentialSmoothing((_joint_position[i] - _joint_position_prev[i]) / elapsed_time.toSec(), _joint_velocity[i], 0.2);
            _joint_effort[i] = _robot_state.getMeasuredTorque()[i];
            // A newly connected or non-commanding FRI session must never retain
            // a stale/zero position target. The trajectory controller may
            // overwrite this after read() once COMMANDING_ACTIVE is reached.
            if (!_commanding || !was_commanding) {
                _joint_position_command[i] = _joint_position[i];
            }
        }
    }

    void Iiwa::_write(ros::Duration elapsed_time)
    {
        if (_idle) // if idle, do nothing
            return;

        // enforce limits
        _position_joint_limits_interface.enforceLimits(elapsed_time);
        _position_joint_saturation_interface.enforceLimits(elapsed_time);
        _effort_joint_limits_interface.enforceLimits(elapsed_time);
        _effort_joint_saturation_interface.enforceLimits(elapsed_time);
        _velocity_joint_limits_interface.enforceLimits(elapsed_time);
        _velocity_joint_saturation_interface.enforceLimits(elapsed_time);

        // reset commmand message
        _fri_message_data->resetCommandMessage();

        if (_robot_state.getClientCommandMode() == kuka::fri::TORQUE) {
            // SafeTorqueController is not itself the authority boundary: any ROS
            // publisher can feed it. The driver therefore sends exactly zero
            // torque unless the heartbeat-backed Demo gate is active.
            if (!_demo_mode_active.load()) {
                std::fill(
                    _joint_effort_command.begin(), _joint_effort_command.end(), 0.0);
            }
            _robot_command.setTorque(_joint_effort_command.data());
            _robot_command.setJointPosition(_joint_position.data());
        }
        else if (_robot_state.getClientCommandMode() == kuka::fri::POSITION) {
            if (_position_command_enabled.load()) {
                const double heartbeat_age = ros::WallTime::now().toSec()
                    - _last_position_heartbeat_wall_sec.load();
                if (heartbeat_age < 0.0 || heartbeat_age > _position_heartbeat_timeout) {
                    for (int i = 0; i < _num_joints; ++i) {
                        _position_hold_command[i] = _joint_position[i];
                    }
                    _position_hold_valid.store(true);
                    _position_arm_requested.store(false);
                    _position_command_enabled.store(false);
                    ROS_ERROR_STREAM_THROTTLE(1.0,
                        "Position executor heartbeat expired after " << heartbeat_age
                        << " s; holding the fixed measured-joint snapshot.");
                }
            }
            if (_position_arm_requested.load() && !_position_command_enabled.load()) {
                double maximum_error = 0.0;
                bool finite = true;
                for (int i = 0; i < _num_joints; ++i) {
                    finite = finite && std::isfinite(_joint_position_command[i]);
                    maximum_error = std::max(maximum_error,
                        std::abs(_joint_position_command[i] - _joint_position[i]));
                }
                if (finite && _commanding.load()
                    && maximum_error <= _position_arm_tolerance) {
                    // Start the timeout at the instant the gate actually opens,
                    // rather than at the start of the blocking service call.
                    _last_position_heartbeat_wall_sec.store(
                        ros::WallTime::now().toSec());
                    _position_command_enabled.store(true);
                    ROS_INFO("FRI position command gate armed after controller synchronization.");
                }
                else {
                    // Keep the request pending until the service-side deadline.
                    // A freshly started JointTrajectoryController may need a few
                    // update cycles to replace its pre-FRI command with the
                    // measured-joint synchronization goal.  Rejecting on the
                    // first cycle made this handshake race-dependent even though
                    // the position gate was still safely closed.
                    ROS_WARN_STREAM_THROTTLE(1.0,
                        "Waiting to arm position commands: synchronized error is "
                        << maximum_error << " rad, commanding="
                        << _commanding.load() << ".");
                }
            }
            if (_position_command_enabled.load())
                _robot_command.setJointPosition(_joint_position_command.data());
            else if (_position_hold_valid.load())
                _robot_command.setJointPosition(_position_hold_command.data());
            else
                // This fallback is used only before the first COMMANDING_ACTIVE
                // sample or during a control-epoch transition.  The control
                // thread latches a fixed hold on that first active sample.
                _robot_command.setJointPosition(_joint_position.data());
        }
        // else ERROR

        _write_fri();
    }

    bool Iiwa::_init_fri()
    {
        _idle = true;
        _commanding = false;

        // Create message/client data
        _fri_message_data = new kuka::fri::ClientData(_robot_state.NUMBER_OF_JOINTS);

        // link monitoring and command message to wrappers
        _robot_state.set_message(&_fri_message_data->monitoringMsg);
        _robot_command.set_message(&_fri_message_data->commandMsg);

        // set specific message IDs
        _fri_message_data->expectedMonitorMsgID = _robot_state.monitoring_message_id();
        _fri_message_data->commandMsg.header.messageIdentifier = _robot_command.command_message_id();

        if (!_connect_fri())
            return false;

        return true;
    }

    bool Iiwa::_connect_fri()
    {
        if (_fri_connection.isOpen()) {
            // TO-DO: Use ROS output
            // printf("Warning: client application already connected!\n");
            return true;
        }

        return _fri_connection.open(_port, _remote_host.c_str());
    }

    void Iiwa::_disconnect_fri()
    {
        if (_fri_connection.isOpen())
            _fri_connection.close();
    }

    bool Iiwa::_read_fri(kuka::fri::ESessionState& current_state)
    {
        if (!_fri_connection.isOpen()) {
            ++_connection_closed_failures;
            _last_io_failure_wall_sec = ros::WallTime::now().toSec();
            ROS_ERROR_THROTTLE(1.0, "FRI receive failed: UDP connection is closed.");
            return false;
        }

        // **************************************************************************
        // Receive and decode new monitoring message
        // **************************************************************************
        _message_size = _fri_connection.receive(_fri_message_data->receiveBuffer, kuka::fri::FRI_MONITOR_MSG_MAX_SIZE);

        if (_message_size <= 0) { // TODO: size == 0 -> connection closed (maybe go to IDLE instead of stopping?)
            ++_receive_failures;
            _last_io_failure_wall_sec = ros::WallTime::now().toSec();
            ROS_ERROR_STREAM_THROTTLE(1.0,
                "FRI receive returned " << _message_size
                << "; receive_failures=" << _receive_failures << ".");
            return false;
        }

        if (!_fri_message_data->decoder.decode(_fri_message_data->receiveBuffer, _message_size)) {
            ++_decode_failures;
            _last_io_failure_wall_sec = ros::WallTime::now().toSec();
            ROS_ERROR_STREAM_THROTTLE(1.0,
                "FRI monitoring message decode failed; decode_failures="
                << _decode_failures << ".");
            return false;
        }

        // check message type (so that our wrappers match)
        if (_fri_message_data->expectedMonitorMsgID != _fri_message_data->monitoringMsg.header.messageIdentifier) {
            ++_message_id_failures;
            _last_io_failure_wall_sec = ros::WallTime::now().toSec();
            ROS_ERROR_STREAM_THROTTLE(1.0,
                "FRI monitoring message ID mismatch: got "
                << _fri_message_data->monitoringMsg.header.messageIdentifier
                << ", expected " << _fri_message_data->expectedMonitorMsgID
                << "; failures=" << _message_id_failures << ".");
            return false;
        }

        current_state = (kuka::fri::ESessionState)_fri_message_data->monitoringMsg.connectionInfo.sessionState;
        _fri_session_state = static_cast<int>(current_state);
        _fri_connection_quality = static_cast<int>(_robot_state.getConnectionQuality());
        _fri_safety_state = static_cast<int>(_robot_state.getSafetyState());
        _fri_operation_mode = static_cast<int>(_robot_state.getOperationMode());
        _fri_drive_state = static_cast<int>(_robot_state.getDriveState());
        _fri_sample_time_sec = _robot_state.getSampleTime();
        _fri_receive_multiplier = static_cast<std::uint32_t>(
            _fri_message_data->monitoringMsg.connectionInfo.receiveMultiplier);

        const std::uint32_t monitor_sequence = static_cast<std::uint32_t>(
            _fri_message_data->monitoringMsg.header.sequenceCounter);
        if (_have_monitor_sequence) {
            const std::uint32_t delta = monitor_sequence - _last_monitor_sequence;
            if (delta == 0) {
                ++_duplicate_monitor_messages;
            }
            else if (delta < std::numeric_limits<std::uint32_t>::max() / 2U) {
                _monitor_sequence_gaps += static_cast<std::uint64_t>(delta - 1U);
            }
            else {
                ++_monitor_sequence_resets;
            }
        }
        _last_monitor_sequence = monitor_sequence;
        _have_monitor_sequence = true;

        if (_fri_message_data->lastState != current_state) {
            _on_fri_state_change(_fri_message_data->lastState, current_state);
            _fri_message_data->lastState = current_state;
        }

        return true;
    }

    bool Iiwa::_write_fri()
    {
        // **************************************************************************
        // Encode and send command message
        // **************************************************************************

        _fri_message_data->lastSendCounter++;
        // check if its time to send an answer
        if (_fri_message_data->lastSendCounter >= _fri_message_data->monitoringMsg.connectionInfo.receiveMultiplier) {
            _fri_message_data->lastSendCounter = 0;

            // set sequence counters
            _fri_message_data->commandMsg.header.sequenceCounter = _fri_message_data->sequenceCounter++;
            _fri_message_data->commandMsg.header.reflectedSequenceCounter = _fri_message_data->monitoringMsg.header.sequenceCounter;

            if (!_fri_message_data->encoder.encode(_fri_message_data->sendBuffer, _message_size)) {
                ++_encode_failures;
                _last_io_failure_wall_sec = ros::WallTime::now().toSec();
                ROS_ERROR_STREAM_THROTTLE(1.0,
                    "FRI command encode failed; encode_failures="
                    << _encode_failures << ".");
                return false;
            }

            if (!_fri_connection.send(_fri_message_data->sendBuffer, _message_size)) {
                ++_send_failures;
                _last_io_failure_wall_sec = ros::WallTime::now().toSec();
                ROS_ERROR_STREAM_THROTTLE(1.0,
                    "FRI command send failed; send_failures="
                    << _send_failures << ".");
                return false;
            }
        }

        return true;
    }
} // namespace iiwa_ros
