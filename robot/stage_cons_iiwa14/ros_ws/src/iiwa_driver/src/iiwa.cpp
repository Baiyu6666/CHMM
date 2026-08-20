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
#include <cmath>
#include <thread>

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
        _last_position_heartbeat_wall_sec.store(0.0);
        _nh = nh;
        _ns = nh.getNamespace();
        _load_params(); // load parameters
        _init(); // initialize
        // add namespace
        _commanding_status_pub = _nh.advertise<std_msgs::Bool>(_ns+"/commanding_status", 100);
        _demo_mode_status_pub = _nh.advertise<std_msgs::Bool>(_ns+"/demo_mode_active", 10, true);
        _client_command_mode_pub = _nh.advertise<std_msgs::Int32>(_ns+"/fri_command_mode", 10, true);
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
        static ros::Rate rate(_control_freq);
        while (ros::ok()) {
            ros::Time time = ros::Time::now();

            // TO-DO: Get real elapsed time?
            auto elapsed_time = ros::Duration(1. / _control_freq);

            _read(elapsed_time);
            _update_demo_motion_gate();
            _controller_manager->update(ros::Time::now(), elapsed_time);
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
            rate.sleep();
        }
    }

    void Iiwa::_publish()
    {
        std_msgs::Bool msg;
        msg.data = _commanding;
        _commanding_status_pub.publish(msg);

        msg.data = _demo_mode_active;
        _demo_mode_status_pub.publish(msg);

        std_msgs::Int32 mode_msg;
        mode_msg.data = _client_command_mode;
        _client_command_mode_pub.publish(mode_msg);
    }

    bool Iiwa::_set_demo_mode(std_srvs::SetBool::Request& request,
        std_srvs::SetBool::Response& response)
    {
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
            _position_command_enabled.store(false);
            response.success = true;
            response.message = "FRI position commands disabled; measured joints are the reference.";
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

        if (requested == _demo_mode_active) {
            return;
        }
        _demo_mode_active = requested;
        ROS_INFO_STREAM("Logical Demo acquisition mode "
                        << (_demo_mode_active ? "enabled" : "disabled")
                        << "; FRI position reference remains measured joints.");
    }

    void Iiwa::_load_params()
    {
        ros::NodeHandle n_p("~");

        n_p.param(_ns + "/iiwa_driver/fri/port", _port, 30200); // Default port is 30200
        n_p.param<std::string>(_ns + "/iiwa_driver/fri/robot_ip", _remote_host, "192.170.10.2"); // Default robot ip is 192.170.10.2
        n_p.param<std::string>(_ns + "/iiwa_driver/fri/robot_description", _robot_description, _ns + "/robot_description");

        n_p.param(_ns + "/iiwa_driver/hardware_interface/control_freq", _control_freq, 200.);
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
        const bool was_commanding = _commanding;
        kuka::fri::ESessionState fri_state = kuka::fri::IDLE;
        if (!_read_fri(fri_state)) {
            _idle = true;
            _commanding = false;
            _client_command_mode = 0;
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
            return;
        }
        _client_command_mode = static_cast<int>(_robot_state.getClientCommandMode());
        if (_client_command_mode != static_cast<int>(kuka::fri::POSITION)) {
            _position_arm_requested.store(false);
            _position_command_enabled.store(false);
        }

        switch (fri_state) {
        case kuka::fri::MONITORING_WAIT:
        case kuka::fri::MONITORING_READY:
        case kuka::fri::COMMANDING_WAIT:
            _idle = false;
            _commanding = false;
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
            _robot_command.setTorque(_joint_effort_command.data());
            _robot_command.setJointPosition(_joint_position.data());
        }
        else if (_robot_state.getClientCommandMode() == kuka::fri::POSITION) {
            if (_position_command_enabled.load()) {
                const double heartbeat_age = ros::WallTime::now().toSec()
                    - _last_position_heartbeat_wall_sec.load();
                if (heartbeat_age < 0.0 || heartbeat_age > _position_heartbeat_timeout) {
                    _position_arm_requested.store(false);
                    _position_command_enabled.store(false);
                    ROS_ERROR_STREAM_THROTTLE(1.0,
                        "Position executor heartbeat expired after " << heartbeat_age
                        << " s; measured joints are the FRI reference.");
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
                if (finite && _commanding && maximum_error <= _position_arm_tolerance) {
                    _position_command_enabled.store(true);
                    ROS_INFO("FRI position command gate armed after controller synchronization.");
                }
                else {
                    _position_arm_requested.store(false);
                    ROS_ERROR_STREAM_THROTTLE(1.0,
                        "Refusing position command arm: synchronized error is "
                        << maximum_error << " rad, commanding=" << _commanding << ".");
                }
            }
            if (_position_command_enabled.load())
                _robot_command.setJointPosition(_joint_position_command.data());
            else
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
            // TO-DO: Use ROS output
            // printf("Error: client application is not connected!\n");
            return false;
        }

        // **************************************************************************
        // Receive and decode new monitoring message
        // **************************************************************************
        _message_size = _fri_connection.receive(_fri_message_data->receiveBuffer, kuka::fri::FRI_MONITOR_MSG_MAX_SIZE);

        if (_message_size <= 0) { // TODO: size == 0 -> connection closed (maybe go to IDLE instead of stopping?)
            // TO-DO: Use ROS output
            // printf("Error: failed while trying to receive monitoring message!\n");
            return false;
        }

        if (!_fri_message_data->decoder.decode(_fri_message_data->receiveBuffer, _message_size)) {
            return false;
        }

        // check message type (so that our wrappers match)
        if (_fri_message_data->expectedMonitorMsgID != _fri_message_data->monitoringMsg.header.messageIdentifier) {
            // TO-DO: Use ROS output
            // printf("Error: incompatible IDs for received message (got: %d expected %d)!\n",
            //     (int)_fri_message_data->monitoringMsg.header.messageIdentifier,
            //     (int)_fri_message_data->expectedMonitorMsgID);
            return false;
        }

        current_state = (kuka::fri::ESessionState)_fri_message_data->monitoringMsg.connectionInfo.sessionState;

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
                return false;
            }

            if (!_fri_connection.send(_fri_message_data->sendBuffer, _message_size)) {
                // TO-DO: Use ROS output
                // printf("Error: failed while trying to send command message!\n");
                return false;
            }
        }

        return true;
    }
} // namespace iiwa_ros
