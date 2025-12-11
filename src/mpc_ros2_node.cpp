/*
 * MIT License
 * MPCRosNode rewritten to use MoveIt for joint references and MPC as the controller.
 * Author: Gabriela
 * TODO: 
 *  - remove hardcoded number of states and controls
 */

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <Eigen/Dense>
#include <map>
#include <vector>
#include <thread>

#include "mpc_ros2/mpc.hpp"

using namespace std::chrono_literals;

namespace MpcRos
{

class MPCRosNode : public rclcpp::Node
{
public:
    MPCRosNode(const std::string & nodeName, const rclcpp::NodeOptions & options);
    ~MPCRosNode();

private:
    // Callbacks
    void displayTrajectoryCallback(const moveit_msgs::msg::DisplayTrajectory::SharedPtr msg);
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    void calculateControl();

    // ROS interfaces
    rclcpp::Subscription<moveit_msgs::msg::DisplayTrajectory>::SharedPtr displayTrajSub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr jointStatesSub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubMpcPath_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pubVacuumCmds_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pubRightArmCmds_;
    rclcpp::TimerBase::SharedPtr controlTimer_;

    // Robot state
    Eigen::VectorXd right_arm_pos_; // 7 DOF
    double vacuum_head_yaw_;
    double vacuum_head_pitch_;

    // Stored joint reference for MPC
    Eigen::VectorXd joint_reference_; // size 9
    bool reference_ready_;

    // Joint names
    std::vector<std::string> rightArmJointNames_;
    std::string headYawJointName_;
    std::string headPitchJointName_;

    // MPC
    std::unique_ptr<MPC<9,9>> mpc_;

    // (Keep for now in case we use move group later)
    std::shared_ptr<rclcpp::Node> moveit_node_;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> moveit_executor_;
    std::thread moveit_spin_thread_;

    // Forward kinematics
    geometry_msgs::msg::Pose computeFK(const std::vector<double>& joint_positions);
    moveit::core::RobotModelPtr moveit_robot_model_;
    moveit::core::RobotStatePtr moveit_robot_state_;
    std::string fk_group_name_ = "right_arm_with_vacuum";
    std::string ee_link_name_ = "vacuum_tip";    

    // State
    bool joint_state_ready_;
};

MPCRosNode::MPCRosNode(const std::string & nodeName, const rclcpp::NodeOptions & options)
: Node(nodeName, options),
  right_arm_pos_(Eigen::VectorXd::Zero(7)),
  vacuum_head_yaw_(0.0), vacuum_head_pitch_(0.0),
  joint_reference_(Eigen::VectorXd::Zero(9)),
  reference_ready_(false), joint_state_ready_(false)
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

    // Subscribe to MoveIt's published planned trajectory (RViz shows this after planning)
    displayTrajSub_ = this->create_subscription<moveit_msgs::msg::DisplayTrajectory>(
        "/display_planned_path", 10,
        std::bind(&MPCRosNode::displayTrajectoryCallback, this, std::placeholders::_1)
    );

    jointStatesSub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10,
        std::bind(&MPCRosNode::jointStateCallback, this, std::placeholders::_1)
    );

    pubMpcPath_ = this->create_publisher<nav_msgs::msg::Path>("/mpc_path", 10);
    pubVacuumCmds_ =
        this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/vacuum_trajectory_controller/joint_trajectory", 10);
    pubRightArmCmds_ =
        this->create_publisher<trajectory_msgs::msg::JointTrajectory>(
            "/right_arm_controller/joint_trajectory", 10);

    controlTimer_ = this->create_wall_timer(100ms, std::bind(&MPCRosNode::calculateControl, this));

    // MPC
    mpc_ = std::make_unique<MPC<9,9>>();

    moveit_node_ = std::make_shared<rclcpp::Node>(
        "moveit_interface_node",
        rclcpp::NodeOptions().append_parameter_override("use_sim_time", true)
    );

    moveit_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    moveit_executor_->add_node(moveit_node_);
    moveit_spin_thread_ = std::thread([this]() { moveit_executor_->spin(); });

    fk_group_name_ = "right_arm_with_vacuum";
    ee_link_name_ = "vacuum_head";    
    robot_model_loader::RobotModelLoader robot_model_loader(
        moveit_node_,
        "robot_description"
    );
    moveit_robot_model_ = robot_model_loader.getModel();
    if (!moveit_robot_model_)
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to load robot model from robot_description!");
        return;
    }

    moveit_robot_state_ = std::make_shared<moveit::core::RobotState>(moveit_robot_model_);
    moveit_robot_state_->setToDefaultValues();
}

MPCRosNode::~MPCRosNode()
{
    moveit_executor_->cancel();
    if (moveit_spin_thread_.joinable())
        moveit_spin_thread_.join();
}

void MPCRosNode::displayTrajectoryCallback(const moveit_msgs::msg::DisplayTrajectory::SharedPtr msg)
{
    if (!msg) return;
    if (msg->trajectory.empty()) return;

    const auto &robot_traj = msg->trajectory.back();
    const auto &jt = robot_traj.joint_trajectory;

    if (jt.points.empty()) 
    {
        RCLCPP_WARN(this->get_logger(), "Received DisplayTrajectory with empty joint_trajectory points; ignoring.");
        return;
    }

    // Build expected joint list in the exact ordering:[headYaw, headPitch, right_arm(0)...right_arm(7)]
    std::vector<std::string> expected_joints;
    expected_joints.push_back(headYawJointName_);
    expected_joints.push_back(headPitchJointName_);
    for (const auto &jn : rightArmJointNames_) expected_joints.push_back(jn);

    // Build name -> index map for incoming trajectory joint_names
    std::map<std::string, size_t> moveit_joints;
    for (size_t i = 0; i < jt.joint_names.size(); ++i) moveit_joints[jt.joint_names[i]] = i;

    // Verify that all expected joints are present in the incoming trajectory (TO MAKE SURE RIGHT MOVE GROUP IS USED)
    for (const auto &jn : expected_joints) {
        if (moveit_joints.find(jn) == moveit_joints.end()) 
        {
            RCLCPP_DEBUG(this->get_logger(),
                         "DisplayTrajectory missing expected joint '%s' — ignoring trajectory.",
                         jn.c_str());
            return;
        }
    }

    const auto &last_point = jt.points.back();
    if (last_point.positions.size() < jt.joint_names.size()) 
    {
        RCLCPP_WARN(this->get_logger(), "DisplayTrajectory last point positions shorter than joint_names; ignoring.");
        return;
    }

    // Fill joint_reference_ in  expected order using mapping by name
    for (size_t i = 0; i < expected_joints.size(); ++i) {
        const auto &jn = expected_joints[i];
        size_t idx = moveit_joints[jn];
        joint_reference_[i] = last_point.positions[idx];
    }

    reference_ready_ = true;

    RCLCPP_INFO(this->get_logger(), "Stored MoveIt/RViz planned joint reference for MPC.");
}

void MPCRosNode::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    if (!msg) return;
    joint_state_ready_ = true;
    for (size_t i = 0; i < msg->name.size(); ++i)
    {
        const auto &n = msg->name[i];
        if (n == headYawJointName_) vacuum_head_yaw_ = msg->position[i];
        else if (n == headPitchJointName_) vacuum_head_pitch_ = msg->position[i];
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

geometry_msgs::msg::Pose MPCRosNode::computeFK(const std::vector<double>& mpc_joint_positions)
{

    moveit_robot_state_->setJointPositions(headYawJointName_, { mpc_joint_positions[0] });
    moveit_robot_state_->setJointPositions(headPitchJointName_, { mpc_joint_positions[1] });
    for (size_t i = 0; i < rightArmJointNames_.size(); ++i)
    {
        moveit_robot_state_->setJointPositions(rightArmJointNames_[i], { mpc_joint_positions[2 + i] });
    }

    moveit_robot_state_->update();

    const Eigen::Isometry3d& ee_tf = moveit_robot_state_->getGlobalLinkTransform(ee_link_name_);

    geometry_msgs::msg::Pose pose;
    pose.position.x = ee_tf.translation().x();
    pose.position.y = ee_tf.translation().y();
    pose.position.z = ee_tf.translation().z();

    Eigen::Quaterniond q(ee_tf.rotation());
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();

    return pose;
}

void MPCRosNode::calculateControl()
{
    if (!joint_state_ready_ || !reference_ready_)
        return;

    // Set references to MPC
    mpc_->set_references(
        joint_reference_[0], joint_reference_[1],
        joint_reference_[2], joint_reference_[3], joint_reference_[4],
        joint_reference_[5], joint_reference_[6], joint_reference_[7],
        joint_reference_[8]
    );

    Eigen::VectorXd state(9);
    state[0] = vacuum_head_yaw_;
    state[1] = vacuum_head_pitch_;
    for (int i = 0; i < 7; i++) state[i + 2] = right_arm_pos_[i];

    RCLCPP_INFO(this->get_logger(), "Current joint state: [%f, %f, %f, %f, %f, %f, %f, %f, %f]",
                state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]);

    RCLCPP_INFO(this->get_logger(), "Joint reference (goal): [%f, %f, %f, %f, %f, %f, %f, %f, %f]",
                joint_reference_[0], joint_reference_[1], joint_reference_[2],
                joint_reference_[3], joint_reference_[4], joint_reference_[5],
                joint_reference_[6], joint_reference_[7], joint_reference_[8]);

    double joint_error = (state - joint_reference_).norm() / std::sqrt(state.size());
    if (joint_error < 0.005) 
    {
        RCLCPP_INFO(this->get_logger(), "Goal REACHED");
        reference_ready_ = false;
        return;
    }


    auto [traj, controls] = mpc_->solve(state);
    if (traj.size() < 2) return;

    std::vector<double> next_positions = traj[1];

    // Publish vacuum trajectory
    trajectory_msgs::msg::JointTrajectory vacuum_msg;
    vacuum_msg.header.stamp = this->now();
    vacuum_msg.joint_names = {headYawJointName_, headPitchJointName_};
    trajectory_msgs::msg::JointTrajectoryPoint vacuum_point;
    vacuum_point.positions = {next_positions[0], next_positions[1]};
    vacuum_point.time_from_start = rclcpp::Duration::from_seconds(0.5);
    vacuum_msg.points.push_back(vacuum_point);
    pubVacuumCmds_->publish(vacuum_msg);

    // Publish arm trajectory
    trajectory_msgs::msg::JointTrajectory arm_msg;
    arm_msg.header.stamp = this->now();
    arm_msg.joint_names = rightArmJointNames_;
    trajectory_msgs::msg::JointTrajectoryPoint arm_point;
    arm_point.positions = {
        next_positions[2], next_positions[3], next_positions[4],
        next_positions[5], next_positions[6], next_positions[7], next_positions[8]
    };
    arm_point.time_from_start = rclcpp::Duration::from_seconds(0.5);
    arm_msg.points.push_back(arm_point);
    pubRightArmCmds_->publish(arm_msg);

    nav_msgs::msg::Path path_msg;
    path_msg.header.frame_id = "base_link"; 
    path_msg.header.stamp = this->now();

    for (const auto& joint_positions : traj)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = "base_link";
        pose_stamped.header.stamp = this->now();

        pose_stamped.pose = computeFK(joint_positions);
        path_msg.poses.push_back(pose_stamped);
    }

    pubMpcPath_->publish(path_msg);
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