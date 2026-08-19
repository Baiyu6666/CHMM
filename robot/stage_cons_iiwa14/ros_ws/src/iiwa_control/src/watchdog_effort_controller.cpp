#include <algorithm>
#include <cmath>
#include <limits>

#include <iiwa_control/watchdog_effort_controller.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <urdf/model.h>

namespace iiwa_control {

WatchdogEffortController::~WatchdogEffortController()
{
    command_subscriber_.shutdown();
}

bool WatchdogEffortController::init(
    hardware_interface::EffortJointInterface* hardware, ros::NodeHandle& node_handle)
{
    if (!node_handle.getParam("joints", joint_names_) || joint_names_.empty()) {
        ROS_ERROR_STREAM("No joints configured in " << node_handle.getNamespace() << ".");
        return false;
    }
    node_handle.param("command_timeout", command_timeout_, 0.05);
    node_handle.param("max_abs_command", max_abs_command_, 2.0);
    node_handle.param("position_stop_margin", position_stop_margin_, 0.0872664626);
    if (!std::isfinite(command_timeout_) || command_timeout_ <= 0.0
        || !std::isfinite(max_abs_command_) || max_abs_command_ < 0.0
        || !std::isfinite(position_stop_margin_) || position_stop_margin_ <= 0.0) {
        ROS_ERROR("Invalid watchdog timeout, effort bound, or joint-position margin.");
        return false;
    }

    std::string robot_description_param;
    std::string robot_description;
    if (!node_handle.searchParam("robot_description", robot_description_param)
        || !node_handle.getParam(robot_description_param, robot_description)) {
        ROS_ERROR("Safe effort controller requires robot_description to enforce position guards.");
        return false;
    }
    urdf::Model model;
    if (!model.initString(robot_description)) {
        ROS_ERROR("Safe effort controller could not parse robot_description.");
        return false;
    }

    try {
        for (const std::string& joint_name : joint_names_) {
            joints_.push_back(hardware->getHandle(joint_name));
            const urdf::JointConstSharedPtr joint = model.getJoint(joint_name);
            if (!joint || !joint->limits) {
                ROS_ERROR_STREAM("Missing URDF position limits for " << joint_name << ".");
                return false;
            }
            if (joint->limits->upper - joint->limits->lower <= 2.0 * position_stop_margin_) {
                ROS_ERROR_STREAM("Position guard margin is too large for " << joint_name << ".");
                return false;
            }
            lower_limits_.push_back(joint->limits->lower);
            upper_limits_.push_back(joint->limits->upper);
        }
    } catch (const hardware_interface::HardwareInterfaceException& error) {
        ROS_ERROR_STREAM("Failed to initialize watchdog effort controller: " << error.what());
        return false;
    }

    TimedCommand initial;
    initial.values.assign(joints_.size(), 0.0);
    initial.valid = false;
    command_buffer_.writeFromNonRT(initial);
    command_subscriber_ = node_handle.subscribe<std_msgs::Float64MultiArray>(
        "command", 1, &WatchdogEffortController::commandCallback, this);
    return true;
}

void WatchdogEffortController::starting(const ros::Time&)
{
    TimedCommand initial;
    initial.values.assign(joints_.size(), 0.0);
    initial.valid = false;
    command_buffer_.writeFromNonRT(initial);
    writeZero();
}

void WatchdogEffortController::update(const ros::Time& time, const ros::Duration&)
{
    if (!withinPositionGuard()) {
        ROS_ERROR_THROTTLE(0.5,
            "SafeTorqueController forced zero: a joint entered the configured limit margin.");
        writeZero();
        return;
    }
    const TimedCommand* command = command_buffer_.readFromRT();
    const double age = command == nullptr ? std::numeric_limits<double>::infinity()
                                          : (time - command->received).toSec();
    const bool fresh = command != nullptr && command->valid && age >= 0.0
        && age <= command_timeout_ && command->values.size() == joints_.size();
    if (!fresh) {
        writeZero();
        return;
    }
    for (std::size_t index = 0; index < joints_.size(); ++index) {
        const double bounded = std::max(
            -max_abs_command_, std::min(command->values[index], max_abs_command_));
        joints_[index].setCommand(bounded);
    }
}

bool WatchdogEffortController::withinPositionGuard() const
{
    for (std::size_t index = 0; index < joints_.size(); ++index) {
        const double position = joints_[index].getPosition();
        if (!std::isfinite(position)
            || position <= lower_limits_[index] + position_stop_margin_
            || position >= upper_limits_[index] - position_stop_margin_) {
            return false;
        }
    }
    return true;
}

void WatchdogEffortController::stopping(const ros::Time&)
{
    writeZero();
}

void WatchdogEffortController::commandCallback(
    const std_msgs::Float64MultiArrayConstPtr& message)
{
    if (message->data.size() != joints_.size()) {
        ROS_ERROR_STREAM_THROTTLE(1.0, "Ignoring effort command of size " << message->data.size()
                                  << "; expected " << joints_.size() << ".");
        return;
    }
    for (double value : message->data) {
        if (!std::isfinite(value)) {
            ROS_ERROR_THROTTLE(1.0, "Ignoring non-finite effort command.");
            return;
        }
    }
    TimedCommand command;
    command.values = message->data;
    command.received = ros::Time::now();
    command.valid = true;
    command_buffer_.writeFromNonRT(command);
}

void WatchdogEffortController::writeZero()
{
    for (hardware_interface::JointHandle& joint : joints_) {
        joint.setCommand(0.0);
    }
}

} // namespace iiwa_control

PLUGINLIB_EXPORT_CLASS(
    iiwa_control::WatchdogEffortController, controller_interface::ControllerBase)
