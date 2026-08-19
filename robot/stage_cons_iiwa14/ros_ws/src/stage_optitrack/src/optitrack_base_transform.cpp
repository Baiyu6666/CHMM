#include <geometry_msgs/PoseStamped.h>
#include <ros/ros.h>
#include <XmlRpcValue.h>

#include <Eigen/Geometry>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool poseToTransform(const geometry_msgs::Pose& pose, Eigen::Isometry3d* transform) {
  const Eigen::Quaterniond quaternion(
      pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
  if (!std::isfinite(quaternion.norm()) || quaternion.norm() < 1e-9 ||
      !std::isfinite(pose.position.x) || !std::isfinite(pose.position.y) ||
      !std::isfinite(pose.position.z)) {
    return false;
  }
  *transform = Eigen::Isometry3d::Identity();
  transform->linear() = quaternion.normalized().toRotationMatrix();
  transform->translation() = Eigen::Vector3d(
      pose.position.x, pose.position.y, pose.position.z);
  return true;
}

geometry_msgs::Pose transformToPose(const Eigen::Isometry3d& transform) {
  geometry_msgs::Pose pose;
  const Eigen::Quaterniond quaternion(transform.linear());
  pose.position.x = transform.translation().x();
  pose.position.y = transform.translation().y();
  pose.position.z = transform.translation().z();
  pose.orientation.x = quaternion.x();
  pose.orientation.y = quaternion.y();
  pose.orientation.z = quaternion.z();
  pose.orientation.w = quaternion.w();
  return pose;
}

class BaseTransformer {
 public:
  BaseTransformer() : private_nh_("~") {
    private_nh_.param<std::string>("base_name", base_name_, "iiwa14");
    private_nh_.param<std::string>("output_frame", output_frame_, "base");
    private_nh_.param("base_timeout", base_timeout_, 0.25);

    XmlRpc::XmlRpcValue names;
    if (!private_nh_.getParam("object_names", names) ||
        names.getType() != XmlRpc::XmlRpcValue::TypeArray || names.size() == 0) {
      throw std::runtime_error("~object_names must be a non-empty string array");
    }

    const std::string prefix = "/vrpn_client_node/";
    base_subscriber_ = nh_.subscribe(
        prefix + base_name_ + "/pose", 10, &BaseTransformer::baseCallback, this);

    for (int index = 0; index < names.size(); ++index) {
      if (names[index].getType() != XmlRpc::XmlRpcValue::TypeString) {
        throw std::runtime_error("Every ~object_names entry must be a string");
      }
      const std::string object_name = static_cast<std::string>(names[index]);
      auto channel = std::unique_ptr<ObjectChannel>(new ObjectChannel);
      channel->name = object_name;
      const std::string input_topic = prefix + object_name + "/pose";
      channel->publisher = nh_.advertise<geometry_msgs::PoseStamped>(
          input_topic + "_from_" + base_name_, 10);
      channel->subscriber = nh_.subscribe<geometry_msgs::PoseStamped>(
          input_topic, 10,
          [this, channel_ptr = channel.get()](const geometry_msgs::PoseStamped::ConstPtr& message) {
            objectCallback(message, channel_ptr);
          });
      channels_.push_back(std::move(channel));
      ROS_INFO_STREAM("OptiTrack object " << object_name << " -> "
                      << input_topic << "_from_" << base_name_);
    }
  }

 private:
  struct ObjectChannel {
    std::string name;
    ros::Subscriber subscriber;
    ros::Publisher publisher;
  };

  void baseCallback(const geometry_msgs::PoseStamped::ConstPtr& message) {
    Eigen::Isometry3d transform;
    if (!poseToTransform(message->pose, &transform)) {
      ROS_WARN_THROTTLE(1.0, "Ignoring invalid OptiTrack base pose");
      return;
    }
    base_in_world_ = transform;
    base_stamp_ = message->header.stamp.isZero() ? ros::Time::now() : message->header.stamp;
    have_base_ = true;
  }

  void objectCallback(const geometry_msgs::PoseStamped::ConstPtr& message,
                      ObjectChannel* channel) {
    if (!have_base_) {
      ROS_WARN_THROTTLE(1.0, "Waiting for OptiTrack base rigid body '%s'", base_name_.c_str());
      return;
    }
    const ros::Time object_stamp =
        message->header.stamp.isZero() ? ros::Time::now() : message->header.stamp;
    if (base_timeout_ > 0.0 && std::abs((object_stamp - base_stamp_).toSec()) > base_timeout_) {
      ROS_WARN_THROTTLE(1.0, "OptiTrack base pose is stale; suppressing relative object pose");
      return;
    }
    Eigen::Isometry3d object_in_world;
    if (!poseToTransform(message->pose, &object_in_world)) {
      ROS_WARN_THROTTLE(1.0, "Ignoring invalid OptiTrack object pose");
      return;
    }
    geometry_msgs::PoseStamped output;
    output.header.stamp = object_stamp;
    output.header.frame_id = output_frame_;
    output.pose = transformToPose(base_in_world_.inverse() * object_in_world);
    channel->publisher.publish(output);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber base_subscriber_;
  std::vector<std::unique_ptr<ObjectChannel>> channels_;
  std::string base_name_;
  std::string output_frame_;
  double base_timeout_ = 0.25;
  bool have_base_ = false;
  ros::Time base_stamp_;
  Eigen::Isometry3d base_in_world_ = Eigen::Isometry3d::Identity();
};

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "optitrack_base_transform");
  try {
    BaseTransformer transformer;
    ros::spin();
  } catch (const std::exception& error) {
    ROS_FATAL_STREAM(error.what());
    return 1;
  }
  return 0;
}
