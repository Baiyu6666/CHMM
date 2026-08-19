#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/jacobian.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>
#include <urdf/model.h>

namespace {

constexpr std::size_t kNumJoints = 7;
constexpr double kEpsilon = 1e-9;

double clampScalar(double value, double lower, double upper)
{
    return std::max(lower, std::min(value, upper));
}

Eigen::Vector3d normalizedOrZero(const Eigen::Vector3d& value)
{
    const double norm = value.norm();
    if (!std::isfinite(norm) || norm <= kEpsilon) {
        return Eigen::Vector3d::Zero();
    }
    return value / norm;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& value)
{
    Eigen::Matrix3d out;
    out << 0.0, -value.z(), value.y(),
        value.z(), 0.0, -value.x(),
        -value.y(), value.x(), 0.0;
    return out;
}

bool finiteVector(const Eigen::VectorXd& value)
{
    return value.array().isFinite().all();
}

bool finiteMatrix(const Eigen::MatrixXd& value)
{
    return value.array().isFinite().all();
}

} // namespace

class DemoVirtualFixture
{
public:
    DemoVirtualFixture(ros::NodeHandle nh, ros::NodeHandle private_nh)
        : nh_(std::move(nh))
        , private_nh_(std::move(private_nh))
    {
    }

    bool init()
    {
        const std::string ns = nh_.getNamespace();
        robot_name_ = ns;
        if (!robot_name_.empty() && robot_name_.front() == '/') {
            robot_name_.erase(robot_name_.begin());
        }
        if (robot_name_.empty()) {
            robot_name_ = "iiwa14";
        }

        private_nh_.param<std::string>("bar_pose_topic", bar_pose_topic_,
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14");
        private_nh_.param<std::string>("expected_bar_frame", expected_bar_frame_, "base");
        private_nh_.param("use_static_bar_axis", use_static_bar_axis_, false);
        private_nh_.param<std::string>("root_link", root_link_, "world");
        private_nh_.param<std::string>("end_effector", end_effector_, robot_name_ + "_link_ee");
        private_nh_.param<std::string>("torque_controller_name", torque_controller_name_, "SafeTorqueController");
        private_nh_.param<std::string>("demo_mode_topic", demo_mode_topic_,
            "/" + robot_name_ + "/demo_mode_active");
        private_nh_.param("demo_mode_timeout", demo_mode_timeout_, 0.5);
        private_nh_.param("control_rate", control_rate_, 200.0);
        private_nh_.param("joint_state_timeout", joint_state_timeout_, 0.10);
        private_nh_.param("bar_pose_timeout", bar_pose_timeout_, 0.20);
        private_nh_.param("bar_axis_filter_alpha", bar_axis_filter_alpha_, 0.15);
        private_nh_.param("twist_filter_alpha", twist_filter_alpha_, 0.15);
        private_nh_.param("orientation_stiffness", orientation_stiffness_, 1.0);
        private_nh_.param("orientation_damping", orientation_damping_, 1.0);
        private_nh_.param("max_orientation_recovery_speed_deg_s",
            max_orientation_recovery_speed_deg_s_, 10.0);
        private_nh_.param("vertical_damping", vertical_damping_, 10.0);
        private_nh_.param("max_orientation_moment", max_orientation_moment_, 0.5);
        private_nh_.param("max_vertical_force", max_vertical_force_, 2.0);
        private_nh_.param("max_joint_assist_torque", max_joint_assist_torque_, 2.0);
        private_nh_.param("max_joint_torque_rate", max_joint_torque_rate_, 5.0);
        private_nh_.param("joint_limit_stop_margin_deg", joint_limit_stop_margin_deg_, 5.0);
        private_nh_.param("enable_ramp_time", enable_ramp_time_, 2.0);
        private_nh_.param("min_tilt_deg", min_tilt_deg_, -80.0);
        private_nh_.param("max_tilt_deg", max_tilt_deg_, 80.0);
        private_nh_.param("start_orientation_enabled", orientation_enabled_, false);
        private_nh_.param("start_vertical_damping_enabled", vertical_damping_enabled_, false);

        bar_axis_local_ = getVectorParam("bar_axis_local", Eigen::Vector3d::UnitX());
        static_bar_axis_base_ = getVectorParam("static_bar_axis_base", Eigen::Vector3d::UnitY());
        table_normal_ = getVectorParam("table_normal_base", Eigen::Vector3d::UnitZ());
        tool_axis_local_ = getVectorParam("tool_axis_local", Eigen::Vector3d::UnitZ());
        tool_lateral_local_ = getVectorParam("tool_lateral_axis_local", Eigen::Vector3d::UnitY());
        tcp_offset_local_ = getVectorParam("tcp_offset_local", Eigen::Vector3d::Zero(), false);

        if (!validateParameters()) {
            return false;
        }

        if (!nh_.getParam(torque_controller_name_ + "/joints", joint_names_)) {
            joint_names_.clear();
            for (std::size_t index = 1; index <= kNumJoints; ++index) {
                joint_names_.push_back(robot_name_ + "_joint_" + std::to_string(index));
            }
            ROS_WARN_STREAM("Could not read " << torque_controller_name_
                            << "/joints; using generated names for " << robot_name_ << ".");
        }
        if (joint_names_.size() != kNumJoints) {
            ROS_ERROR_STREAM("Expected " << kNumJoints << " controlled joints, got " << joint_names_.size() << ".");
            return false;
        }
        for (std::size_t index = 0; index < joint_names_.size(); ++index) {
            joint_index_[joint_names_[index]] = index;
        }

        std::string robot_description_param;
        if (!nh_.searchParam("robot_description", robot_description_param)) {
            ROS_ERROR("Could not find robot_description on the ROS parameter server.");
            return false;
        }
        std::string urdf_string;
        if (!nh_.getParam(robot_description_param, urdf_string) || urdf_string.empty()) {
            ROS_ERROR_STREAM("Could not read URDF from " << robot_description_param << ".");
            return false;
        }
        urdf::Model urdf_model;
        if (!urdf_model.initString(urdf_string)) {
            ROS_ERROR("Could not parse robot_description for joint limits.");
            return false;
        }
        for (const std::string& joint_name : joint_names_) {
            const urdf::JointConstSharedPtr joint = urdf_model.getJoint(joint_name);
            if (!joint || !joint->limits) {
                ROS_ERROR_STREAM("Missing URDF limits for " << joint_name << ".");
                return false;
            }
            joint_lower_limits_.push_back(joint->limits->lower);
            joint_upper_limits_.push_back(joint->limits->upper);
        }

        KDL::Tree tree;
        if (!kdl_parser::treeFromString(urdf_string, tree)
            || !tree.getChain(root_link_, end_effector_, kinematic_chain_)
            || kinematic_chain_.getNrOfJoints() != kNumJoints) {
            ROS_ERROR_STREAM("Could not build a 7-joint KDL chain from " << root_link_
                             << " to " << end_effector_ << ".");
            return false;
        }
        fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(kinematic_chain_));
        jacobian_solver_.reset(new KDL::ChainJntToJacSolver(kinematic_chain_));

        if (use_static_bar_axis_) {
            static_bar_axis_base_ -= table_normal_ * table_normal_.dot(static_bar_axis_base_);
            static_bar_axis_base_ = normalizedOrZero(static_bar_axis_base_);
            if (static_bar_axis_base_.isZero(kEpsilon)) {
                ROS_ERROR("static_bar_axis_base must not be parallel to table_normal_base.");
                return false;
            }
            filtered_bar_axis_ = static_bar_axis_base_;
            bar_pose_received_ = true;
            bar_pose_arrival_ = ros::Time::now();
        }

        joint_position_.setZero(kNumJoints);
        joint_velocity_.setZero(kNumJoints);
        last_command_.setZero(kNumJoints);

        joint_state_sub_ = nh_.subscribe<sensor_msgs::JointState>(
            "joint_states", 1, &DemoVirtualFixture::jointStateCallback, this,
            ros::TransportHints().reliable().tcpNoDelay());
        demo_mode_sub_ = nh_.subscribe<std_msgs::Bool>(
            demo_mode_topic_, 1, &DemoVirtualFixture::demoModeCallback, this,
            ros::TransportHints().reliable().tcpNoDelay());
        if (!use_static_bar_axis_) {
            bar_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
                bar_pose_topic_, 1, &DemoVirtualFixture::barPoseCallback, this,
                ros::TransportHints().reliable().tcpNoDelay());
        }

        torque_pub_ = nh_.advertise<std_msgs::Float64MultiArray>(torque_controller_name_ + "/command", 1);
        status_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("demo_virtual_fixture/status", 1);
        bar_axis_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("demo_virtual_fixture/bar_axis", 1);
        bar_lateral_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("demo_virtual_fixture/bar_lateral", 1);
        tool_axis_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>("demo_virtual_fixture/tool_axis", 1);
        desired_tool_axis_pub_ = nh_.advertise<geometry_msgs::Vector3Stamped>(
            "demo_virtual_fixture/desired_tool_axis", 1);

        orientation_enable_service_ = private_nh_.advertiseService(
            "enable_orientation", &DemoVirtualFixture::enableOrientation, this);
        vertical_enable_service_ = private_nh_.advertiseService(
            "enable_vertical_damping", &DemoVirtualFixture::enableVerticalDamping, this);
        all_enable_service_ = private_nh_.advertiseService(
            "enable_all", &DemoVirtualFixture::enableAll, this);
        reload_tuning_service_ = private_nh_.advertiseService(
            "reload_tuning", &DemoVirtualFixture::reloadTuning, this);

        ROS_INFO_STREAM("Demo virtual fixture initialized for /" << robot_name_
                        << ". Output: /" << robot_name_ << "/" << torque_controller_name_ << "/command");
        if (use_static_bar_axis_) {
            ROS_WARN_STREAM("Using static bar direction in robot base: "
                            << static_bar_axis_base_.transpose()
                            << ". OptiTrack bar poses are intentionally ignored.");
        } else {
            ROS_INFO_STREAM("Bar pose input: " << bar_pose_topic_ << ", expected frame: "
                            << expected_bar_frame_ << ", local bar axis: " << bar_axis_local_.transpose());
        }
        ROS_WARN("Assistance starts disabled unless explicitly enabled in the config. Verify axes before enabling torque.");
        return true;
    }

    void run()
    {
        ros::Rate rate(control_rate_);
        last_update_time_ = ros::Time::now();
        while (ros::ok()) {
            ros::spinOnce();
            const ros::Time now = ros::Time::now();
            const double dt = clampScalar((now - last_update_time_).toSec(), 0.0, 0.05);
            last_update_time_ = now;
            update(now, dt);
            rate.sleep();
        }
        for (int attempt = 0; attempt < 10; ++attempt) {
            publishZeroCommand();
            ros::WallDuration(0.005).sleep();
        }
    }

private:
    Eigen::Vector3d getVectorParam(
        const std::string& name, const Eigen::Vector3d& default_value, bool normalize = true)
    {
        std::vector<double> values;
        Eigen::Vector3d result = default_value;
        if (private_nh_.getParam(name, values)) {
            if (values.size() != 3) {
                ROS_ERROR_STREAM("Parameter " << private_nh_.resolveName(name) << " must contain three values.");
                parameter_error_ = true;
                return result;
            }
            result << values[0], values[1], values[2];
        }
        if (!result.array().isFinite().all()) {
            ROS_ERROR_STREAM("Parameter " << private_nh_.resolveName(name) << " contains a non-finite value.");
            parameter_error_ = true;
            return default_value;
        }
        if (normalize) {
            result = normalizedOrZero(result);
            if (result.isZero(kEpsilon)) {
                ROS_ERROR_STREAM("Parameter " << private_nh_.resolveName(name) << " must be nonzero.");
                parameter_error_ = true;
                return default_value;
            }
        }
        return result;
    }

    bool validateParameters()
    {
        if (parameter_error_) {
            return false;
        }
        table_normal_ = normalizedOrZero(table_normal_);
        bar_axis_local_ = normalizedOrZero(bar_axis_local_);
        tool_axis_local_ = normalizedOrZero(tool_axis_local_);

        tool_lateral_local_ -= tool_axis_local_ * tool_axis_local_.dot(tool_lateral_local_);
        tool_lateral_local_ = normalizedOrZero(tool_lateral_local_);
        if (tool_lateral_local_.isZero(kEpsilon)) {
            ROS_ERROR("tool_lateral_axis_local must not be parallel to tool_axis_local.");
            return false;
        }
        tool_third_local_ = normalizedOrZero(tool_axis_local_.cross(tool_lateral_local_));
        local_tool_frame_.col(0) = tool_axis_local_;
        local_tool_frame_.col(1) = tool_lateral_local_;
        local_tool_frame_.col(2) = tool_third_local_;

        if (!(control_rate_ > 0.0) || !(joint_state_timeout_ > 0.0) || !(bar_pose_timeout_ > 0.0)
            || !(max_joint_assist_torque_ >= 0.0) || !(max_joint_torque_rate_ >= 0.0)
            || !(max_orientation_moment_ >= 0.0) || !(max_vertical_force_ >= 0.0)
            || !(orientation_stiffness_ >= 0.0) || !(orientation_damping_ > 0.0)
            || !(max_orientation_recovery_speed_deg_s_ > 0.0)
            || !(joint_limit_stop_margin_deg_ > 0.0)
            || !(demo_mode_timeout_ > 0.0)
            || !(vertical_damping_ >= 0.0) || !(enable_ramp_time_ >= 0.0)) {
            ROS_ERROR("Virtual fixture gains, limits, rates, and timeouts are invalid; orientation damping and recovery speed must be positive.");
            return false;
        }
        bar_axis_filter_alpha_ = clampScalar(bar_axis_filter_alpha_, 0.0, 1.0);
        twist_filter_alpha_ = clampScalar(twist_filter_alpha_, 0.0, 1.0);
        if (!(min_tilt_deg_ < max_tilt_deg_)) {
            ROS_ERROR("min_tilt_deg must be smaller than max_tilt_deg.");
            return false;
        }
        min_tilt_rad_ = min_tilt_deg_ * M_PI / 180.0;
        max_tilt_rad_ = max_tilt_deg_ * M_PI / 180.0;
        max_orientation_recovery_speed_rad_s_
            = max_orientation_recovery_speed_deg_s_ * M_PI / 180.0;
        joint_limit_stop_margin_rad_ = joint_limit_stop_margin_deg_ * M_PI / 180.0;
        return true;
    }

    void jointStateCallback(const sensor_msgs::JointState::ConstPtr& msg)
    {
        if (msg->position.size() != msg->name.size() || msg->velocity.size() != msg->name.size()) {
            ROS_WARN_THROTTLE(1.0, "Ignoring joint_states without aligned position and velocity arrays.");
            return;
        }

        Eigen::VectorXd position(kNumJoints);
        Eigen::VectorXd velocity(kNumJoints);
        std::vector<bool> found(kNumJoints, false);
        for (std::size_t msg_index = 0; msg_index < msg->name.size(); ++msg_index) {
            const auto match = joint_index_.find(msg->name[msg_index]);
            if (match == joint_index_.end()) {
                continue;
            }
            const std::size_t index = match->second;
            position[index] = msg->position[msg_index];
            velocity[index] = msg->velocity[msg_index];
            found[index] = true;
        }
        if (!std::all_of(found.begin(), found.end(), [](bool value) { return value; })
            || !finiteVector(position) || !finiteVector(velocity)) {
            ROS_WARN_THROTTLE(1.0, "Ignoring incomplete or non-finite iiwa joint state.");
            return;
        }
        joint_position_ = position;
        joint_velocity_ = velocity;
        joint_state_received_ = true;
        joint_state_arrival_ = ros::Time::now();
    }

    void demoModeCallback(const std_msgs::Bool::ConstPtr& msg)
    {
        demo_mode_active_ = msg->data;
        demo_mode_received_ = true;
        demo_mode_arrival_ = ros::Time::now();
    }

    bool demoModeFreshAndActive() const
    {
        return demo_mode_received_ && demo_mode_active_
            && (ros::Time::now() - demo_mode_arrival_).toSec() <= demo_mode_timeout_;
    }

    void barPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        if (use_static_bar_axis_) {
            return;
        }
        if (!expected_bar_frame_.empty() && msg->header.frame_id != expected_bar_frame_) {
            ROS_ERROR_STREAM_THROTTLE(1.0, "Rejecting bar pose in frame '" << msg->header.frame_id
                                    << "'; expected '" << expected_bar_frame_ << "'.");
            return;
        }
        const Eigen::Quaterniond quaternion(
            msg->pose.orientation.w, msg->pose.orientation.x,
            msg->pose.orientation.y, msg->pose.orientation.z);
        if (!std::isfinite(quaternion.norm()) || quaternion.norm() <= kEpsilon) {
            ROS_WARN_THROTTLE(1.0, "Ignoring invalid OptiTrack bar quaternion.");
            return;
        }
        const Eigen::Matrix3d rotation = quaternion.normalized().toRotationMatrix();
        bar_axis_candidates_ = rotation;

        Eigen::Vector3d candidate = rotation * bar_axis_local_;
        candidate -= table_normal_ * table_normal_.dot(candidate);
        candidate = normalizedOrZero(candidate);
        if (candidate.isZero(kEpsilon)) {
            ROS_ERROR_THROTTLE(1.0, "Configured bar axis is nearly parallel to the table normal.");
            return;
        }
        if (bar_pose_received_ && filtered_bar_axis_.dot(candidate) < 0.0) {
            candidate = -candidate;
        }
        if (!bar_pose_received_ || bar_axis_filter_alpha_ >= 1.0) {
            filtered_bar_axis_ = candidate;
        } else {
            filtered_bar_axis_ = normalizedOrZero(
                (1.0 - bar_axis_filter_alpha_) * filtered_bar_axis_ + bar_axis_filter_alpha_ * candidate);
        }
        if (filtered_bar_axis_.isZero(kEpsilon)) {
            return;
        }
        bar_pose_received_ = true;
        bar_pose_arrival_ = ros::Time::now();
        bar_pose_stamp_ = msg->header.stamp;
    }

    bool enableOrientation(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
    {
        if (request.data && !demoModeFreshAndActive()) {
            response.success = false;
            response.message = "Orientation assistance rejected: Demo collection mode is inactive or stale.";
            return true;
        }
        if (request.data && !jointLimitsSafe()) {
            response.success = false;
            response.message = "Orientation assistance rejected: joint state is missing or inside the limit guard margin.";
            return true;
        }
        if (request.data && !orientation_enabled_) {
            assistance_ramp_ = 0.0;
        }
        orientation_enabled_ = request.data;
        resetRampAndCommandIfDisabled();
        response.success = true;
        response.message = orientation_enabled_ ? "Orientation fixture requested." : "Orientation fixture disabled.";
        return true;
    }

    bool enableVerticalDamping(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
    {
        if (request.data && !demoModeFreshAndActive()) {
            response.success = false;
            response.message = "Vertical assistance rejected: Demo collection mode is inactive or stale.";
            return true;
        }
        if (request.data && !jointLimitsSafe()) {
            response.success = false;
            response.message = "Vertical assistance rejected: joint state is missing or inside the limit guard margin.";
            return true;
        }
        vertical_damping_enabled_ = request.data;
        resetRampAndCommandIfDisabled();
        response.success = true;
        response.message = vertical_damping_enabled_ ? "Vertical damping requested." : "Vertical damping disabled.";
        return true;
    }

    bool enableAll(std_srvs::SetBool::Request& request, std_srvs::SetBool::Response& response)
    {
        if (request.data && !demoModeFreshAndActive()) {
            response.success = false;
            response.message = "Assistance rejected: Demo collection mode is inactive or stale.";
            return true;
        }
        if (request.data && !jointLimitsSafe()) {
            response.success = false;
            response.message = "Assistance rejected: joint state is missing or inside the limit guard margin.";
            return true;
        }
        if (request.data && !orientation_enabled_) {
            assistance_ramp_ = 0.0;
        }
        orientation_enabled_ = request.data;
        vertical_damping_enabled_ = request.data;
        resetRampAndCommandIfDisabled();
        response.success = true;
        response.message = request.data ? "All virtual fixtures requested." : "All virtual fixtures disabled.";
        return true;
    }

    bool reloadTuning(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& response)
    {
        if (orientation_enabled_ || vertical_damping_enabled_) {
            response.success = false;
            response.message = "Disable both assistance channels before reloading tuning.";
            return true;
        }

        double orientation_stiffness = orientation_stiffness_;
        double orientation_damping = orientation_damping_;
        double recovery_speed_deg_s = max_orientation_recovery_speed_deg_s_;
        double vertical_damping = vertical_damping_;
        double max_orientation_moment = max_orientation_moment_;
        double max_vertical_force = max_vertical_force_;
        double max_joint_assist_torque = max_joint_assist_torque_;
        double max_joint_torque_rate = max_joint_torque_rate_;
        double enable_ramp_time = enable_ramp_time_;

        const bool complete
            = private_nh_.getParam("orientation_stiffness", orientation_stiffness)
            && private_nh_.getParam("orientation_damping", orientation_damping)
            && private_nh_.getParam("max_orientation_recovery_speed_deg_s", recovery_speed_deg_s)
            && private_nh_.getParam("vertical_damping", vertical_damping)
            && private_nh_.getParam("max_orientation_moment", max_orientation_moment)
            && private_nh_.getParam("max_vertical_force", max_vertical_force)
            && private_nh_.getParam("max_joint_assist_torque", max_joint_assist_torque)
            && private_nh_.getParam("max_joint_torque_rate", max_joint_torque_rate)
            && private_nh_.getParam("enable_ramp_time", enable_ramp_time);
        if (!complete) {
            response.success = false;
            response.message = "One or more tuning parameters are missing; previous tuning retained.";
            return true;
        }

        const bool finite = std::isfinite(orientation_stiffness)
            && std::isfinite(orientation_damping) && std::isfinite(recovery_speed_deg_s)
            && std::isfinite(vertical_damping) && std::isfinite(max_orientation_moment)
            && std::isfinite(max_vertical_force) && std::isfinite(max_joint_assist_torque)
            && std::isfinite(max_joint_torque_rate) && std::isfinite(enable_ramp_time);
        // These are tuning guard rails, not certified robot safety limits.
        const bool in_range = orientation_stiffness >= 0.0 && orientation_stiffness <= 10.0
            && orientation_damping > 0.0 && orientation_damping <= 10.0
            && recovery_speed_deg_s > 0.0 && recovery_speed_deg_s <= 30.0
            // High anisotropic damping must become perceptible during slow
            // kinesthetic motion. Physical output remains independently bounded
            // by the Cartesian-force, joint-torque, torque-rate, and ramp limits.
            && vertical_damping >= 0.0 && vertical_damping <= 2500.0
            && max_orientation_moment >= 0.0 && max_orientation_moment <= 1.0
            && max_vertical_force >= 0.0 && max_vertical_force <= 25.0
            && max_joint_assist_torque >= 0.0 && max_joint_assist_torque <= 8.0
            && max_joint_torque_rate > 0.0 && max_joint_torque_rate <= 10.0
            && enable_ramp_time >= 0.5 && enable_ramp_time <= 10.0;
        if (!finite || !in_range) {
            response.success = false;
            response.message = "Tuning rejected: a value is non-finite or outside the conservative reload range.";
            return true;
        }

        orientation_stiffness_ = orientation_stiffness;
        orientation_damping_ = orientation_damping;
        max_orientation_recovery_speed_deg_s_ = recovery_speed_deg_s;
        max_orientation_recovery_speed_rad_s_ = recovery_speed_deg_s * M_PI / 180.0;
        vertical_damping_ = vertical_damping;
        max_orientation_moment_ = max_orientation_moment;
        max_vertical_force_ = max_vertical_force;
        max_joint_assist_torque_ = max_joint_assist_torque;
        max_joint_torque_rate_ = max_joint_torque_rate;
        enable_ramp_time_ = enable_ramp_time;
        assistance_ramp_ = 0.0;
        publishZeroCommand();

        response.success = true;
        response.message = "Assistance tuning reloaded; both channels remain disabled.";
        ROS_INFO_STREAM("Reloaded assistance tuning: orientation stiffness=" << orientation_stiffness_
                        << ", damping=" << orientation_damping_
                        << ", moment limit=" << max_orientation_moment_
                        << ", vertical damping=" << vertical_damping_
                        << ", force limit=" << max_vertical_force_);
        return true;
    }

    void resetRampAndCommandIfDisabled()
    {
        if (!orientation_enabled_ && !vertical_damping_enabled_) {
            assistance_ramp_ = 0.0;
            publishZeroCommand();
        }
    }

    double minimumJointLimitMargin() const
    {
        if (!joint_state_received_) {
            return -std::numeric_limits<double>::infinity();
        }
        double minimum_margin = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < kNumJoints; ++index) {
            minimum_margin = std::min(minimum_margin,
                std::min(joint_position_[index] - joint_lower_limits_[index],
                    joint_upper_limits_[index] - joint_position_[index]));
        }
        return minimum_margin;
    }

    bool jointLimitsSafe() const
    {
        return joint_state_received_
            && (ros::Time::now() - joint_state_arrival_).toSec() <= joint_state_timeout_
            && minimumJointLimitMargin() > joint_limit_stop_margin_rad_;
    }

    void update(const ros::Time& now, double dt)
    {
        const double joint_age = joint_state_received_
            ? (now - joint_state_arrival_).toSec()
            : std::numeric_limits<double>::infinity();
        const double bar_age = use_static_bar_axis_
            ? 0.0
            : (bar_pose_received_ ? (now - bar_pose_arrival_).toSec()
                                  : std::numeric_limits<double>::infinity());
        const bool inputs_valid = joint_state_received_ && bar_pose_received_
            && joint_age <= joint_state_timeout_ && bar_age <= bar_pose_timeout_;
        const bool assistance_requested = orientation_enabled_ || vertical_damping_enabled_;
        const bool demo_gate_active = demoModeFreshAndActive();

        if (!demo_gate_active) {
            if (assistance_requested) {
                ROS_ERROR_THROTTLE(1.0,
                    "Virtual fixture disabled because Demo collection mode is inactive or stale.");
            }
            orientation_enabled_ = false;
            vertical_damping_enabled_ = false;
            assistance_ramp_ = 0.0;
            publishZeroCommand();
            publishStatus(now, joint_age, bar_age, 0.0, 0.0, 0.0, 0.0, false,
                minimumJointLimitMargin(), false);
            return;
        }

        const bool requested = assistance_requested;

        if (!inputs_valid) {
            if (requested) {
                ROS_WARN_STREAM_THROTTLE(1.0, "Virtual fixture output is zero: joint age=" << joint_age
                                         << " s, bar age=" << bar_age << " s.");
            }
            assistance_ramp_ = 0.0;
            publishZeroCommand();
            publishStatus(now, joint_age, bar_age, 0.0, 0.0, 0.0, 0.0, false,
                minimumJointLimitMargin(), false);
            return;
        }
        const double minimum_joint_margin = minimumJointLimitMargin();
        if (minimum_joint_margin <= joint_limit_stop_margin_rad_) {
            if (requested) {
                ROS_ERROR_STREAM_THROTTLE(0.5,
                    "Virtual fixture disabled at joint-limit guard; minimum margin="
                    << minimum_joint_margin * 180.0 / M_PI << " deg.");
            }
            orientation_enabled_ = false;
            vertical_damping_enabled_ = false;
            assistance_ramp_ = 0.0;
            publishZeroCommand();
            publishStatus(now, joint_age, bar_age, 0.0, 0.0, 0.0, 0.0, false,
                minimum_joint_margin, true);
            return;
        }
        if (!requested) {
            assistance_ramp_ = 0.0;
        }

        KDL::JntArray positions(kNumJoints);
        for (std::size_t index = 0; index < kNumJoints; ++index) {
            positions(index) = joint_position_[index];
        }

        KDL::Frame ee_frame;
        KDL::Jacobian kdl_jacobian(kNumJoints);
        if (fk_solver_->JntToCart(positions, ee_frame) < 0
            || jacobian_solver_->JntToJac(positions, kdl_jacobian) < 0
            || !finiteMatrix(kdl_jacobian.data)) {
            ROS_ERROR_THROTTLE(1.0, "Non-finite or malformed FK/Jacobian; publishing zero assist torque.");
            assistance_ramp_ = 0.0;
            publishZeroCommand();
            return;
        }

        Eigen::Matrix3d rotation;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                rotation(row, column) = ee_frame.M(row, column);
            }
        }
        const Eigen::MatrixXd angular_jacobian = kdl_jacobian.data.bottomRows(3);
        const Eigen::Vector3d tcp_offset_world = rotation * tcp_offset_local_;
        const Eigen::MatrixXd linear_jacobian
            = kdl_jacobian.data.topRows(3) - skew(tcp_offset_world) * angular_jacobian;

        const Eigen::Vector3d raw_angular_velocity = angular_jacobian * joint_velocity_;
        const Eigen::Vector3d raw_linear_velocity = linear_jacobian * joint_velocity_;
        if (!twist_initialized_) {
            filtered_angular_velocity_ = raw_angular_velocity;
            filtered_linear_velocity_ = raw_linear_velocity;
            twist_initialized_ = true;
        } else {
            filtered_angular_velocity_ = (1.0 - twist_filter_alpha_) * filtered_angular_velocity_
                + twist_filter_alpha_ * raw_angular_velocity;
            filtered_linear_velocity_ = (1.0 - twist_filter_alpha_) * filtered_linear_velocity_
                + twist_filter_alpha_ * raw_linear_velocity;
        }

        const Eigen::Vector3d bar_axis = filtered_bar_axis_;
        const Eigen::Vector3d bar_lateral = normalizedOrZero(table_normal_.cross(bar_axis));
        if (bar_lateral.isZero(kEpsilon)) {
            publishZeroCommand();
            return;
        }

        const Eigen::Vector3d tool_axis = normalizedOrZero(rotation * tool_axis_local_);
        const double raw_tilt = std::atan2(tool_axis.dot(bar_axis), -tool_axis.dot(table_normal_));
        const double desired_tilt = clampScalar(raw_tilt, min_tilt_rad_, max_tilt_rad_);
        const Eigen::Vector3d desired_tool_axis = normalizedOrZero(
            -std::cos(desired_tilt) * table_normal_ + std::sin(desired_tilt) * bar_axis);

        Eigen::Matrix3d world_tool_frame;
        world_tool_frame.col(0) = desired_tool_axis;
        world_tool_frame.col(1) = bar_lateral;
        world_tool_frame.col(2) = normalizedOrZero(desired_tool_axis.cross(bar_lateral));
        const Eigen::Matrix3d desired_rotation = world_tool_frame * local_tool_frame_.transpose();

        Eigen::AngleAxisd angle_axis(desired_rotation * rotation.transpose());
        Eigen::Vector3d orientation_error = Eigen::Vector3d::Zero();
        if (std::isfinite(angle_axis.angle()) && angle_axis.axis().array().isFinite().all()) {
            orientation_error = angle_axis.angle() * angle_axis.axis();
        }

        const Eigen::Matrix3d forbidden_rotation = Eigen::Matrix3d::Identity()
            - bar_lateral * bar_lateral.transpose();
        const bool tilt_at_boundary = raw_tilt < min_tilt_rad_ || raw_tilt > max_tilt_rad_;
        if (!tilt_at_boundary) {
            orientation_error = forbidden_rotation * orientation_error;
        }
        const Eigen::Vector3d forbidden_angular_velocity
            = forbidden_rotation * filtered_angular_velocity_;

        Eigen::Vector3d orientation_moment = Eigen::Vector3d::Zero();
        if (orientation_enabled_) {
            Eigen::Vector3d desired_recovery_velocity
                = (orientation_stiffness_ / orientation_damping_) * orientation_error;
            const double desired_speed = desired_recovery_velocity.norm();
            if (desired_speed > max_orientation_recovery_speed_rad_s_
                && desired_speed > kEpsilon) {
                desired_recovery_velocity *= max_orientation_recovery_speed_rad_s_ / desired_speed;
            }
            orientation_moment = orientation_damping_
                * (desired_recovery_velocity - forbidden_angular_velocity);
            const double moment_norm = orientation_moment.norm();
            if (moment_norm > max_orientation_moment_ && moment_norm > kEpsilon) {
                orientation_moment *= max_orientation_moment_ / moment_norm;
            }
        }

        // iiwa_driver already filters the differentiated joint velocities. Use
        // that current signal directly for the high-gain vertical channel. A
        // second low-pass filter adds enough phase lag that the damper can keep
        // the old sign after a reversal and then be repeatedly suppressed by
        // the passivity guard.
        const double vertical_velocity = table_normal_.dot(raw_linear_velocity);
        double vertical_force = 0.0;
        if (vertical_damping_enabled_) {
            vertical_force = clampScalar(
                -vertical_damping_ * vertical_velocity, -max_vertical_force_, max_vertical_force_);
        }

        Eigen::VectorXd command = angular_jacobian.transpose() * orientation_moment
            + linear_jacobian.transpose() * (vertical_force * table_normal_);
        if (!finiteVector(command)) {
            ROS_ERROR_THROTTLE(1.0, "Computed non-finite assist torque; publishing zero.");
            publishZeroCommand();
            return;
        }

        if (requested) {
            if (enable_ramp_time_ <= kEpsilon) {
                assistance_ramp_ = 1.0;
            } else {
                assistance_ramp_ = std::min(1.0, assistance_ramp_ + dt / enable_ramp_time_);
            }
        } else {
            command.setZero();
        }
        command *= assistance_ramp_;
        // Scale the complete joint-torque vector instead of clipping joints
        // independently. Independent clipping changes the J^T wrench direction
        // and makes a nominally vertical fixture feel like an arbitrary joint
        // resistance near saturation.
        const double peak_torque = command.cwiseAbs().maxCoeff();
        if (peak_torque > max_joint_assist_torque_ && peak_torque > kEpsilon) {
            command *= max_joint_assist_torque_ / peak_torque;
        }

        // Apply the slew limit to the complete delta for the same reason: every
        // joint advances by one common scale factor toward the desired vector.
        Eigen::VectorXd slew_origin = last_command_;
        if (vertical_damping_enabled_ && !orientation_enabled_
            && slew_origin.dot(joint_velocity_) > 0.0) {
            // On a direction reversal, the previous damping torque now points
            // with the motion. Release it immediately and slew in from zero in
            // the new dissipative direction.
            slew_origin.setZero();
        }
        const Eigen::VectorXd command_delta = command - slew_origin;
        const double peak_delta = command_delta.cwiseAbs().maxCoeff();
        const double max_delta = max_joint_torque_rate_ * dt;
        if (peak_delta > max_delta && peak_delta > kEpsilon) {
            command = slew_origin + command_delta * (max_delta / peak_delta);
        }

        // For the pure vertical mode, numerical/Jacobian changes must never turn
        // the commanded torque into positive mechanical power. Orientation
        // recovery is excluded because its spring term intentionally restores
        // pose and is not a pure damper.
        if (vertical_damping_enabled_ && !orientation_enabled_
            && command.dot(joint_velocity_) > 1e-9) {
            command.setZero();
        }

        publishCommand(command);
        publishAxes(now, bar_axis, bar_lateral, tool_axis, desired_tool_axis);
        publishStatus(now, joint_age, bar_age, vertical_velocity, raw_tilt,
            orientation_error.norm(), vertical_force, requested, minimum_joint_margin, false);

        if (requested) {
            ROS_INFO_STREAM_THROTTLE(1.0,
                "fixture active ori=" << orientation_enabled_
                << " z_damping=" << vertical_damping_enabled_
                << " tilt_deg=" << raw_tilt * 180.0 / M_PI
                << " ori_err_deg=" << orientation_error.norm() * 180.0 / M_PI
                << " vz=" << vertical_velocity
                << " tau_norm=" << command.norm());
        }
        if (!use_static_bar_axis_) {
            ROS_INFO_STREAM_THROTTLE(5.0,
                "OptiTrack bar local axes in base: x=[" << bar_axis_candidates_.col(0).transpose()
                << "] y=[" << bar_axis_candidates_.col(1).transpose()
                << "] z=[" << bar_axis_candidates_.col(2).transpose() << "]");
        }
    }

    void publishCommand(const Eigen::VectorXd& command)
    {
        std_msgs::Float64MultiArray message;
        message.data.resize(kNumJoints);
        for (std::size_t index = 0; index < kNumJoints; ++index) {
            message.data[index] = command[index];
        }
        torque_pub_.publish(message);
        last_command_ = command;
    }

    void publishZeroCommand()
    {
        if (!torque_pub_) {
            return;
        }
        Eigen::VectorXd zero = Eigen::VectorXd::Zero(kNumJoints);
        publishCommand(zero);
    }

    void publishAxes(const ros::Time& stamp, const Eigen::Vector3d& bar_axis,
        const Eigen::Vector3d& bar_lateral, const Eigen::Vector3d& tool_axis,
        const Eigen::Vector3d& desired_tool_axis)
    {
        publishVector(bar_axis_pub_, stamp, bar_axis);
        publishVector(bar_lateral_pub_, stamp, bar_lateral);
        publishVector(tool_axis_pub_, stamp, tool_axis);
        publishVector(desired_tool_axis_pub_, stamp, desired_tool_axis);
    }

    void publishVector(ros::Publisher& publisher, const ros::Time& stamp, const Eigen::Vector3d& value)
    {
        geometry_msgs::Vector3Stamped message;
        message.header.stamp = stamp;
        message.header.frame_id = expected_bar_frame_;
        message.vector.x = value.x();
        message.vector.y = value.y();
        message.vector.z = value.z();
        publisher.publish(message);
    }

    void publishStatus(const ros::Time&, double joint_age, double bar_age,
        double vertical_velocity, double tilt, double orientation_error,
        double vertical_force, bool active, double minimum_joint_margin,
        bool joint_limit_guard_active)
    {
        std_msgs::Float64MultiArray message;
        message.data = {
            active ? 1.0 : 0.0,
            orientation_enabled_ ? 1.0 : 0.0,
            vertical_damping_enabled_ ? 1.0 : 0.0,
            joint_age,
            bar_age,
            vertical_velocity,
            tilt,
            orientation_error,
            vertical_force,
            last_command_.norm(),
            assistance_ramp_,
            minimum_joint_margin,
            joint_limit_guard_active ? 1.0 : 0.0,
            demoModeFreshAndActive() ? 1.0 : 0.0,
        };
        status_pub_.publish(message);
    }

    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Subscriber joint_state_sub_;
    ros::Subscriber demo_mode_sub_;
    ros::Subscriber bar_pose_sub_;
    ros::Publisher torque_pub_;
    ros::Publisher status_pub_;
    ros::Publisher bar_axis_pub_;
    ros::Publisher bar_lateral_pub_;
    ros::Publisher tool_axis_pub_;
    ros::Publisher desired_tool_axis_pub_;
    ros::ServiceServer orientation_enable_service_;
    ros::ServiceServer vertical_enable_service_;
    ros::ServiceServer all_enable_service_;
    ros::ServiceServer reload_tuning_service_;

    KDL::Chain kinematic_chain_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jacobian_solver_;
    std::string robot_name_;
    std::string bar_pose_topic_;
    std::string expected_bar_frame_;
    std::string root_link_;
    std::string end_effector_;
    std::string demo_mode_topic_;
    std::string torque_controller_name_;
    std::vector<std::string> joint_names_;
    std::vector<double> joint_lower_limits_;
    std::vector<double> joint_upper_limits_;
    std::unordered_map<std::string, std::size_t> joint_index_;

    Eigen::VectorXd joint_position_;
    Eigen::VectorXd joint_velocity_;
    Eigen::VectorXd last_command_;
    Eigen::Vector3d bar_axis_local_ = Eigen::Vector3d::UnitX();
    Eigen::Vector3d static_bar_axis_base_ = Eigen::Vector3d::UnitY();
    Eigen::Vector3d table_normal_ = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d tool_axis_local_ = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d tool_lateral_local_ = Eigen::Vector3d::UnitY();
    Eigen::Vector3d tool_third_local_ = Eigen::Vector3d::UnitX();
    Eigen::Vector3d tcp_offset_local_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d local_tool_frame_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d bar_axis_candidates_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d filtered_bar_axis_ = Eigen::Vector3d::UnitX();
    Eigen::Vector3d filtered_angular_velocity_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d filtered_linear_velocity_ = Eigen::Vector3d::Zero();

    ros::Time joint_state_arrival_;
    ros::Time demo_mode_arrival_;
    ros::Time bar_pose_arrival_;
    ros::Time bar_pose_stamp_;
    ros::Time last_update_time_;
    bool parameter_error_ = false;
    bool use_static_bar_axis_ = false;
    bool joint_state_received_ = false;
    bool demo_mode_received_ = false;
    bool demo_mode_active_ = false;
    bool bar_pose_received_ = false;
    bool twist_initialized_ = false;
    bool orientation_enabled_ = false;
    bool vertical_damping_enabled_ = false;
    double assistance_ramp_ = 0.0;

    double control_rate_ = 200.0;
    double joint_state_timeout_ = 0.10;
    double demo_mode_timeout_ = 0.5;
    double bar_pose_timeout_ = 0.20;
    double bar_axis_filter_alpha_ = 0.15;
    double twist_filter_alpha_ = 0.15;
    double orientation_stiffness_ = 1.0;
    double orientation_damping_ = 1.0;
    double max_orientation_recovery_speed_deg_s_ = 10.0;
    double max_orientation_recovery_speed_rad_s_ = 10.0 * M_PI / 180.0;
    double vertical_damping_ = 10.0;
    double max_orientation_moment_ = 0.5;
    double max_vertical_force_ = 2.0;
    double max_joint_assist_torque_ = 2.0;
    double max_joint_torque_rate_ = 5.0;
    double joint_limit_stop_margin_deg_ = 5.0;
    double joint_limit_stop_margin_rad_ = 5.0 * M_PI / 180.0;
    double enable_ramp_time_ = 2.0;
    double min_tilt_deg_ = -80.0;
    double max_tilt_deg_ = 80.0;
    double min_tilt_rad_ = -80.0 * M_PI / 180.0;
    double max_tilt_rad_ = 80.0 * M_PI / 180.0;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "demo_virtual_fixture");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    DemoVirtualFixture fixture(nh, private_nh);
    if (!fixture.init()) {
        return 1;
    }
    fixture.run();
    return 0;
}
