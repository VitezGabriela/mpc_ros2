/*
 * MIT License
 * Adapted for robot-arm-pushed vacuum with head joints
 * Author: Mustafa Ege Kural (original)
 * Adaptation: Gabriela Vitez
 * TODO:
 *  - fix hardcoded bounds, make it load from urdf
 */

#include <cstddef>
#include <vector>
#include <tuple>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <Eigen/Dense>
#include <cassert>
#include <iostream>
#include "mpc_ros2/mpc.hpp"

namespace MpcRos
{

/**
 * @brief Templated FG_eval 
 *
 * State vector: x ∈ R^nSstates
 * Control vector: u ∈ R^nControls
 */
template<int nStates, int nControls>
class FG_eval
{
public:
    // Template parameters 
    static constexpr int n_states_   = nStates;
    static constexpr int n_controls_ = nControls;

    // MPC parameters
    std::vector<double> ref_joints_;   // reference trajectory
    int mpc_horizon_;
    double dt_;

    std::vector<double> w_joints_;     // weights
    double w_control_;
    double w_control_change_;
    double w_terminal_;

    int joint_start_;
    int joint_rate_start_;

    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

    FG_eval(const std::vector<double>& ref_joints, int horizon = 20, double dt = 0.1)
        : ref_joints_(ref_joints), mpc_horizon_(horizon), dt_(dt)
    {
        assert(ref_joints_.size() == n_states_);

        w_joints_ = {10, 10, 50, 50, 50, 10, 50, 10, 10};
        w_control_ = 10.0;
        w_control_change_ = 1.0;
        w_terminal_ = 50.0;

        joint_start_ = 0;
        joint_rate_start_ = n_states_ * mpc_horizon_;
    }

    void operator()(ADvector& fg, const ADvector& vars)
    {
        const size_t n_constraints = n_states_ + (mpc_horizon_ - 1) * n_states_;

        const size_t expected_fg_size = 1 + n_constraints;
        if (fg.size() != expected_fg_size) fg.resize(expected_fg_size);

        // Cost
        fg[0] = 0;

        // State tracking cost
        for (int t = 0; t < mpc_horizon_ - 1; ++t) 
        {
            for (int j = 0; j < n_states_; ++j) 
            {
                size_t idx = joint_start_ + t * n_states_ + j;
                fg[0] += w_joints_[j] * CppAD::pow(vars[idx] - ref_joints_[j], 2);
            }
        }

        // Control effort cost
        for (int t = 0; t < mpc_horizon_ - 1; ++t) 
        {
            for (int j = 0; j < n_controls_; ++j) 
            {
                size_t idx = joint_rate_start_ + t * n_controls_ + j;
                fg[0] += w_control_ * CppAD::pow(vars[idx], 2);
            }
        }

        // Smoothness cost
        for (int t = 0; t < mpc_horizon_ - 2; ++t) 
        {
            for (int j = 0; j < n_controls_; ++j) 
            {
                size_t idx0 = joint_rate_start_ + t * n_controls_ + j;
                size_t idx1 = joint_rate_start_ + (t + 1) * n_controls_ + j;
                fg[0] += w_control_change_ * CppAD::pow(vars[idx1] - vars[idx0], 2);
            }
        }

        // Terminal cost
        for (int j = 0; j < n_states_; ++j) 
        {
            size_t idx = joint_start_ + (mpc_horizon_ - 1) * n_states_ + j;
            fg[0] += w_terminal_ * CppAD::pow(vars[idx] - ref_joints_[j], 2);
        }

        // Initial state constraint
        for (int j = 0; j < n_states_; ++j) 
        {
            fg[1 + j] = vars[joint_start_ + j];
        }

        // Dynamics: x_{t+1} = x_t + u_t * dt
        const size_t offset = 1 + n_states_;
        for (int t = 0; t < mpc_horizon_ - 1; ++t) 
        {
            for (int j = 0; j < n_states_; ++j) 
            {
                size_t idx_fg = offset + t * n_states_ + j;
                size_t idx_x0 = joint_start_ + t * n_states_ + j;
                size_t idx_x1 = joint_start_ + (t + 1) * n_states_ + j;
                size_t idx_u  = joint_rate_start_ + t * n_controls_ + j;

                fg[idx_fg] = vars[idx_x1] - (vars[idx_x0] + vars[idx_u] * dt_);
            }
        }
    }
};


/**
 * @brief MPC solver class 
 */
template<int nStates, int nControls>
MPC<nStates, nControls>::MPC()
{
    mpc_horizon_ = 20;
    max_rate_ = 5.0;
    bound_value_ = 1.0e3;

    references_.resize(nStates, 0.0);
    last_controls_.assign(nControls, 0.0);
}
template<int nStates, int nControls>
void MPC<nStates, nControls>::set_references(double j0, double j1, double j2, double j3, double j4, double j5, double j6, double j7, double j8)
{
    references_ = {j0, j1, j2, j3, j4, j5, j6, j7, j8};
}

template<int nStates, int nControls>
std::tuple<std::vector<std::vector<double>>, std::vector<double>> MPC<nStates, nControls>::solve(const Eigen::VectorXd& state)
{
    typedef CPPAD_TESTVECTOR(double) Dvector;

    const int H = mpc_horizon_;
    // number of decision variables
    const size_t n_vars = (size_t)H * nStates + (size_t)(H - 1) * nControls;
    // number of constraints
    const size_t n_constraints = nStates + (size_t)(H - 1) * nStates;

    Dvector vars(n_vars), vars_lowerbound(n_vars), vars_upperbound(n_vars);
    Dvector constraints_lowerbound(n_constraints), constraints_upperbound(n_constraints);

    // Initial guess
    for (int t = 0; t < H; ++t) 
    {
        double alpha = t / double(H - 1);
        for (int j = 0; j < nStates; ++j) 
        {
            vars[t * nStates + j] = (1 - alpha) * state[j] + alpha * references_[j];
        }
    }

    // HARDCODED BOUNDS -FIX
    const std::vector<double> joint_min = {-2*M_PI/3, 0, -3.1, -2.0, -3.1, -1.8, -3.1, -3.1, -1.7};
    const std::vector<double> joint_max = {2*M_PI/3, M_PI/2, 3.1,  2.0,  3.1,  1.8,  3.1,  3.1,  1.7};
    const std::vector<double> joint_vel_max = {3.0, 3.0, M_PI, M_PI, 3.92699, 3.92699, 3.92699, 3.92699, 3.92699};
    // State bounds
    for (int i = 0; i < nStates*H; ++i) 
    {
        int j = i % nStates;
        vars_lowerbound[i] = joint_min[j];
        vars_upperbound[i] = joint_max[j];
    }
    // Control bounds
    for (size_t i = static_cast<size_t>(nStates) * H; i < n_vars; ++i)
    {
        size_t j = (i - static_cast<size_t>(nStates) * H) % static_cast<size_t>(nControls);
        vars_lowerbound[i] = -joint_vel_max[j];
        vars_upperbound[i] = joint_vel_max[j];
    }

    // Constraints
    for (size_t i = 0; i < n_constraints; ++i)
        constraints_lowerbound[i] = constraints_upperbound[i] = 0.0;

    // Initial state equality, rewrite constraints
    for (int j = 0; j < nStates; ++j) 
    {
        constraints_lowerbound[j] = state[j];
        constraints_upperbound[j] = state[j];
        vars[j] = state[j];  // warm start
    }

    // Warm-start controls
    const size_t control_start = (size_t)nStates * (size_t)H;
    const size_t control_steps = (size_t)(H - 1);
    for (size_t t = 0; t < control_steps; ++t)
    {
        for (int j = 0; j < nControls; ++j)
        {
            size_t idx = control_start + t * nControls + j;
            vars[idx] = last_controls_[j];
        }
    }

    // Solve
    FG_eval<nStates, nControls> fg_eval(references_, mpc_horizon_, 0.1);
    CppAD::ipopt::solve_result<Dvector> solution;

    std::string printOpt;
    printOpt += "Integer print_level 5\n";
    printOpt += "Sparse true forward\n";
    printOpt += "Sparse true reverse\n";
    printOpt += "String linear_solver mumps\n";
    printOpt += "Numeric max_cpu_time 0.5\n";

    CppAD::ipopt::solve<Dvector, FG_eval<nStates, nControls>>(
        printOpt,
        vars, vars_lowerbound, vars_upperbound,
        constraints_lowerbound, constraints_upperbound,
        fg_eval, solution);

    bool ok = (solution.status == CppAD::ipopt::solve_result<Dvector>::success);
    if (!ok)
        std::cerr << "MPC solve failed!\n";

    // Safety fallback 
    if (solution.x.size() < n_vars) 
    {
        std::cerr << "[MPC] WARNING: Bad solution size fallback\n";
        std::vector<std::vector<double>> traj(H, std::vector<double>(nStates));
        for (int t = 0; t < H; ++t)
            for (int j = 0; j < nStates; ++j)
                traj[t][j] = state[j];
        return { traj, last_controls_ };
    }

    // Extract trajectory
    std::vector<std::vector<double>> traj(H, std::vector<double>(nStates));
    for (int t = 0; t < H; ++t)
        for (int j = 0; j < nStates; ++j)
            traj[t][j] = solution.x[t * nStates + j];

    // Extract first control
    std::vector<double> controls(nControls, 0.0);
    const size_t first_control_idx = (size_t)nStates * (size_t)H;
    if (first_control_idx + nControls <= solution.x.size())
    {
        for (int j = 0; j < nControls; ++j)
            controls[j] = solution.x[first_control_idx + j];
    }

    // Save first controls for next warm-start
    last_controls_ = controls;

    return { traj, controls };
}
template class MpcRos::MPC<9, 9>;
} // namespace MpcRos
