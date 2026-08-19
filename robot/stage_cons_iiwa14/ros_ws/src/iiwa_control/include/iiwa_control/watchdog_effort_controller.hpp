#ifndef IIWA_CONTROL_WATCHDOG_EFFORT_CONTROLLER_HPP
#define IIWA_CONTROL_WATCHDOG_EFFORT_CONTROLLER_HPP

#include <string>
#include <vector>

#include <controller_interface/controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <std_msgs/Float64MultiArray.h>

namespace iiwa_control {

class WatchdogEffortController
    : public controller_interface::Controller<hardware_interface::EffortJointInterface>
{
public:
    WatchdogEffortController() = default;
    ~WatchdogEffortController() override;

    bool init(hardware_interface::EffortJointInterface* hardware, ros::NodeHandle& node_handle) override;
    void starting(const ros::Time& time) override;
    void update(const ros::Time& time, const ros::Duration& period) override;
    void stopping(const ros::Time& time) override;

private:
    struct TimedCommand
    {
        std::vector<double> values;
        ros::Time received;
        bool valid = false;
    };

    void commandCallback(const std_msgs::Float64MultiArrayConstPtr& message);
    void writeZero();
    bool withinPositionGuard() const;

    std::vector<hardware_interface::JointHandle> joints_;
    std::vector<std::string> joint_names_;
    std::vector<double> lower_limits_;
    std::vector<double> upper_limits_;
    realtime_tools::RealtimeBuffer<TimedCommand> command_buffer_;
    ros::Subscriber command_subscriber_;
    double command_timeout_ = 0.05;
    double max_abs_command_ = 2.0;
    double position_stop_margin_ = 0.0872664626;
};

} // namespace iiwa_control

#endif
