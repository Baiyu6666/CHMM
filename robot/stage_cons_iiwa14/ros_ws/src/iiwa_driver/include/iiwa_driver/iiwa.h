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
#ifndef IIWA_DRIVER_IIWA_H
#define IIWA_DRIVER_IIWA_H

#include <atomic>
#include <cstdint>

// ROS Headers
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <std_srvs/SetBool.h>

#include <realtime_tools/realtime_publisher.h>

#include <iiwa_driver/AdditionalOutputs.h>
#include <iiwa_driver/FriDiagnostics.h>
#include <std_msgs/Float64MultiArray.h>

#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/joint_state_interface.h>
#include <hardware_interface/robot_hw.h>

#include <joint_limits_interface/joint_limits.h>
#include <joint_limits_interface/joint_limits_interface.h>
#include <joint_limits_interface/joint_limits_rosparam.h>
#include <joint_limits_interface/joint_limits_urdf.h>

// FRI Headers
#include <kuka/fri/LBRCommand.h>
#include <kuka/fri/LBRState.h>
#include <kuka/fri/UdpConnection.h>

namespace controller_manager {
    class ControllerManager;
}

namespace kuka {
    namespace fri {
        class ClientData;

        class DummyState : public LBRState {
        public:
            FRIMonitoringMessage* message() { return _message; }
            void set_message(FRIMonitoringMessage* msg) { _message = msg; }
            int monitoring_message_id() { return LBRMONITORMESSAGEID; }
        };

        class DummyCommand : public LBRCommand {
        public:
            FRICommandMessage* message() { return _message; }
            void set_message(FRICommandMessage* msg) { _message = msg; }
            int command_message_id() { return LBRCOMMANDMESSAGEID; }
        };
    } // namespace fri
} // namespace kuka

namespace iiwa_ros {
    class Iiwa : public hardware_interface::RobotHW {
    public:
        Iiwa(ros::NodeHandle& nh);
        ~Iiwa();

        void init(ros::NodeHandle& nh);
        void run();
        bool initialized();

    protected:
        void _init();
        void _ctrl_loop();
        bool _configure_control_thread();
        void _load_params();
        void _read(ros::Duration elapsed_time);
        void _write(ros::Duration elapsed_time);
        bool _init_fri();
        bool _connect_fri();
        void _disconnect_fri();
        bool _read_fri(kuka::fri::ESessionState& current_state);
        bool _write_fri();
        void _publish();
        void _publish_fri_diagnostics();
        void _update_demo_motion_gate();
        bool _set_demo_mode(std_srvs::SetBool::Request& request,
            std_srvs::SetBool::Response& response);
        bool _set_position_commanding(std_srvs::SetBool::Request& request,
            std_srvs::SetBool::Response& response);
        void _demo_heartbeat(const std_msgs::Empty::ConstPtr& message);
        void _position_command_heartbeat(const std_msgs::Empty::ConstPtr& message);
        void _on_fri_state_change(
            kuka::fri::ESessionState old_state, kuka::fri::ESessionState current_state);

        // External torque publisher
        realtime_tools::RealtimePublisher<iiwa_driver::AdditionalOutputs> _additional_pub;
        realtime_tools::RealtimePublisher<iiwa_driver::FriDiagnostics> _fri_diagnostics_pub;

        // Interfaces
        hardware_interface::JointStateInterface _joint_state_interface;
        hardware_interface::PositionJointInterface _position_joint_interface;
        hardware_interface::VelocityJointInterface _velocity_joint_interface;
        hardware_interface::EffortJointInterface _effort_joint_interface;

        joint_limits_interface::EffortJointSaturationInterface _effort_joint_saturation_interface;
        joint_limits_interface::EffortJointSoftLimitsInterface _effort_joint_limits_interface;
        joint_limits_interface::PositionJointSaturationInterface _position_joint_saturation_interface;
        joint_limits_interface::PositionJointSoftLimitsInterface _position_joint_limits_interface;
        joint_limits_interface::VelocityJointSaturationInterface _velocity_joint_saturation_interface;
        joint_limits_interface::VelocityJointSoftLimitsInterface _velocity_joint_limits_interface;

        // Shared memory
        int _num_joints;
        int _joint_mode; // position, velocity, or effort
        std::vector<std::string> _joint_names;
        std::vector<int> _joint_types;
        std::vector<double> _joint_position, _joint_position_prev;
        std::vector<double> _joint_velocity;
        std::vector<double> _joint_effort;
        std::vector<double> _joint_position_command;
        // Fixed fail-closed reference used while POSITION FRI remains active
        // but no executor owns the command gate.  This must not be replaced by
        // the changing measured position on every cycle: doing so creates a
        // moving reference and can let the arm drift after Stop Execution.
        std::vector<double> _position_hold_command;
        std::vector<double> _joint_velocity_command;
        std::vector<double> _joint_effort_command;

        // Controller manager
        std::shared_ptr<controller_manager::ControllerManager> _controller_manager;

        // FRI Connection
        kuka::fri::UdpConnection _fri_connection;
        kuka::fri::ClientData* _fri_message_data;
        kuka::fri::DummyState _robot_state; //!< wrapper class for the FRI monitoring message
        kuka::fri::DummyCommand _robot_command; //!< wrapper class for the FRI command message
        int _message_size;
        bool _idle;
        std::atomic<bool> _commanding;
        std::atomic<int> _client_command_mode;

        // FRI diagnostics are updated only by the control thread and published
        // at low rate through a realtime publisher. They never gate commands.
        std::uint64_t _control_cycles;
        std::uint64_t _cycle_period_deadline_misses;
        std::uint64_t _control_work_overruns;
        double _last_cycle_start_wall_sec;
        double _last_cycle_period_sec;
        double _maximum_cycle_period_sec;
        double _last_control_work_sec;
        double _maximum_control_work_sec;
        double _last_deadline_miss_wall_sec;
        double _fri_diagnostics_publish_period;
        double _fri_deadline_miss_factor;
        double _last_fri_diagnostics_publish_wall_sec;
        int _fri_realtime_priority;
        int _fri_cpu_affinity;
        bool _fri_realtime_enabled;
        int _fri_realtime_effective_priority;
        int _fri_effective_cpu_affinity;

        std::uint64_t _connection_closed_failures;
        std::uint64_t _receive_failures;
        std::uint64_t _decode_failures;
        std::uint64_t _message_id_failures;
        std::uint64_t _encode_failures;
        std::uint64_t _send_failures;
        std::uint64_t _monitor_sequence_gaps;
        std::uint64_t _duplicate_monitor_messages;
        std::uint64_t _monitor_sequence_resets;
        std::uint32_t _last_monitor_sequence;
        bool _have_monitor_sequence;
        double _last_io_failure_wall_sec;

        int _fri_session_state;
        int _fri_connection_quality;
        int _fri_safety_state;
        int _fri_operation_mode;
        int _fri_drive_state;
        double _fri_sample_time_sec;
        std::uint32_t _fri_receive_multiplier;
        std::uint64_t _fri_session_state_changes;
        double _last_fri_session_state_change_wall_sec;

        // Demo state gates acquisition and optional assistance only. It never
        // changes the FRI joint-position reference.
        std::atomic<bool> _demo_mode_requested;
        std::atomic<double> _last_demo_heartbeat_wall_sec;
        std::atomic<bool> _demo_mode_active;
        double _demo_heartbeat_timeout;
        std::atomic<bool> _position_command_enabled;
        std::atomic<bool> _position_arm_requested;
        std::atomic<std::uint64_t> _position_controller_reset_requested;
        std::atomic<std::uint64_t> _position_controller_reset_completed;
        std::atomic<bool> _position_hold_valid;
        std::atomic<double> _last_position_heartbeat_wall_sec;
        double _position_arm_tolerance;
        double _position_heartbeat_timeout;

        int _port;
        std::string _remote_host;

        // ROS communication/timing related
        ros::NodeHandle _nh;
        std::string _ns;
        std::string _robot_description;
        ros::Duration _control_period;
        realtime_tools::RealtimePublisher<std_msgs::Bool> _commanding_status_pub;
        realtime_tools::RealtimePublisher<std_msgs::Bool> _demo_mode_status_pub;
        realtime_tools::RealtimePublisher<std_msgs::Int32> _client_command_mode_pub;
        ros::ServiceServer _position_command_service;
        ros::Subscriber _position_heartbeat_sub;
        ros::Subscriber _demo_heartbeat_sub;
        ros::ServiceServer _demo_mode_service;
        double _control_freq;
        double _status_publish_period;
        double _last_status_publish_wall_sec;
        bool _initialized;
    };
} // namespace iiwa_ros

#endif
