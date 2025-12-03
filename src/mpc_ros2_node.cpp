/*
 * MIT License
 * MPCRosNode rewritten to use MoveIt for joint references and MPC as the controller.
 * Author: Gabriela
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/path.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>

#include "mpc_ros2/mpc.hpp"
#include <Eigen/Dense>
#include <vector>

using namespace std::chrono_literals;

namespace MpcRos
{

class MPCRosNode : public rclcpp::Node
{
public:
    MPCRosNode(const std::string & nodeName, const rclcpp::NodeOptions & options);

private:
    void localizationCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
    void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void calculateControl();

    // Subscribers / Publishers / Timer
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goalSub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr jointStatesSub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubMpcPath_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pubVacuumCmds_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pubRightArmCmds_;
    rclcpp::TimerBase::SharedPtr controlTimer_;

    // Robot state
    Eigen::VectorXd right_arm_pos_; // 7 DOF
    double vacuum_head_yaw_;
    double vacuum_head_pitch_;

    // Goal
    Eigen::Vector3d goal_pos_;
    bool goal_received_;

    // Joint names
    std::vector<std::string> rightArmJointNames_;
    std::string headYawJointName_;
    std::string headPitchJointName_;

    // MPC and MoveIt
    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::unique_ptr<MPC> mpc_;
};

MPCRosNode::MPCRosNode(const std::string & nodeName, const rclcpp::NodeOptions & options)
: Node(nodeName, options),
  right_arm_pos_(Eigen::VectorXd::Zero(7)),
  vacuum_head_yaw_(0.0), vacuum_head_pitch_(0.0),
  goal_pos_(Eigen::Vector3d::Zero()), goal_received_(false)
{
    // Parameters
    this->declare_parameter<std::string>("vacuum_head_yaw_joint", "vacuum_body_to_stick_root");
    this->declare_parameter<std::string>("vacuum_head_pitch_joint", "vacuum_stick_root_to_head");
    this->get_parameter("vacuum_head_yaw_joint", headYawJointName_);
    this->get_parameter("vacuum_head_pitch_joint", headPitchJointName_);

    rightArmJointNames_ = {
        "right_shoulder_y", "right_shoulder_x", "right_shoulder_z",
        "right_elbow_y", "right_wrist_z", "right_wrist_x", "right_wrist_y"
    };

    // Subscribers
    goalSub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal", 10, std::bind(&MPCRosNode::goalCallback, this, std::placeholders::_1));
    jointStatesSub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10, std::bind(&MPCRosNode::jointStateCallback, this, std::placeholders::_1));

    // Publishers
    pubMpcPath_ = this->create_publisher<nav_msgs::msg::Path>("/mpc_path", 10);
    pubVacuumCmds_ = this->create_publisher<sensor_msgs::msg::JointState>("/vacuum_trajectory_controller/commands", 10);
    pubRightArmCmds_ = this->create_publisher<sensor_msgs::msg::JointState>("/right_arm_controller/commands", 10);

    // Control timer
    controlTimer_ = this->create_wall_timer(100ms, std::bind(&MPCRosNode::calculateControl, this));

    // MPC
    mpc_ = std::make_unique<MPC>();
}

void MPCRosNode::goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    goal_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
    goal_received_ = true;
}

void MPCRosNode::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    for (size_t i = 0; i < msg->name.size(); ++i)
    {
        const auto &n = msg->name[i];
        if (n == headYawJointName_) vacuum_head_yaw_ = msg->position[i];
        else if (n == headPitchJointName_) vacuum_head_pitch_ = msg->position[i]; // passive
        else
        {
            for (size_t j = 0; j < rightArmJointNames_.size(); ++j)
            {
                if (n == rightArmJointNames_[j])
                    right_arm_pos_[j] = msg->position[i];
            }
        }
    }
}

void MPCRosNode::calculateControl()
{
    if (!goal_received_) return;

    if (!move_group_) {
        move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
            this->shared_from_this(),
            "right_arm_with_vacuum"
        );
    }

    moveit::core::RobotStatePtr kinematic_state = move_group_->getCurrentState();
    const moveit::core::JointModelGroup* joint_group = kinematic_state->getJointModelGroup("vacuum_and_right_arm");

    geometry_msgs::msg::PoseStamped goal_pose;
    goal_pose.header.frame_id = move_group_->getPlanningFrame();
    goal_pose.pose.position.x = goal_pos_.x();
    goal_pose.pose.position.y = goal_pos_.y();
    goal_pose.pose.position.z = goal_pos_.z();
    goal_pose.pose.orientation.w = 1.0;

    bool ik_found = kinematic_state->setFromIK(joint_group, goal_pose.pose);
    if (!ik_found)
    {
        RCLCPP_WARN(this->get_logger(), "MoveIt IK failed for goal");
        return;
    }

    // Copy joint positions for MPC reference
    std::vector<double> joint_references;
    kinematic_state->copyJointGroupPositions(joint_group, joint_references);

    if (joint_references.size() != 9)
    {
        RCLCPP_WARN(this->get_logger(), "Unexpected number of joints from MoveIt IK");
        return;
    }

    // Push references into MPC (order must match)
    mpc_->set_references(
        joint_references[0], // vacuum yaw
        joint_references[1], // vacuum pitch
        joint_references[2], // right_shoulder_y
        joint_references[3], // right_shoulder_x
        joint_references[4], // right_shoulder_z
        joint_references[5], // right_elbow_y
        joint_references[6], // right_wrist_z
        joint_references[7], // right_wrist_x
        joint_references[8]  // right_wrist_y
    );

    Eigen::VectorXd state(9);

    // vacuum joints
    state[0] = vacuum_head_yaw_;
    state[1] = vacuum_head_pitch_;

    // right arm joints
    for (size_t i = 0; i < 7; ++i)
        state[i + 2] = right_arm_pos_[i];

    // Solve MPC
    auto [traj, controls] = mpc_->solve(state);
    if (traj.size() < 2) return;

    // Next predicted positions
    std::vector<double> next_positions = traj[1];

    // --- Publish vacuum joints ---
    sensor_msgs::msg::JointState vacuum_cmd;
    vacuum_cmd.header.stamp = this->now();
    vacuum_cmd.name = {"vacuum_body_to_stick_root", "vacuum_stick_root_to_head"};
    vacuum_cmd.position = {next_positions[0], next_positions[1]};
    pubVacuumCmds_->publish(vacuum_cmd);

    // --- Publish right arm joints ---
    sensor_msgs::msg::JointState right_cmd;
    right_cmd.header.stamp = this->now();
    right_cmd.name = {
        "right_shoulder_y", "right_shoulder_x", "right_shoulder_z",
        "right_elbow_y", "right_wrist_z", "right_wrist_x", "right_wrist_y"
    };
    right_cmd.position = {
        next_positions[2], next_positions[3], next_positions[4],
        next_positions[5], next_positions[6], next_positions[7], next_positions[8]
    };
    pubRightArmCmds_->publish(right_cmd);
}

} // namespace MpcRos

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions opts;
    opts.automatically_declare_parameters_from_overrides(true);
    auto node = std::make_shared<MpcRos::MPCRosNode>("mpc_ros2_node", opts);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
