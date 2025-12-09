/*
 * MIT License
 * Adapted for robot-arm-pushed vacuum with head joints
 * Author: Mustafa Ege Kural (original)
 * Adaptation: Gabriela Vitez
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
 * @brief FG_eval class for MPC
 *
 * State vector: [joint1..joint9]
 * Control vector: [joint1_rate..joint9_rate]
 */
class FG_eval
{
public:
    // MPC parameters
    int mpc_horizon_;
    double dt_;
    static constexpr int n_states_ = 9;
    static constexpr int n_controls_ = 9;
    
    std::vector<double> ref_joints_;   // reference trajectory

    std::vector<double> w_joints_;     // weights
    double w_control_;
    double w_control_change_;
    double w_terminal_;

    int joint_start_;
    int joint_rate_start_;

    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

    FG_eval(const std::vector<double>& ref_joints,
            int horizon = 20,
            double dt = 0.1)
        : ref_joints_(ref_joints), mpc_horizon_(horizon), dt_(dt)
    {
        assert(ref_joints_.size() == n_states_);

        w_joints_ = {10, 10, 50, 50, 50, 10, 50, 10, 10};
        w_control_ = 5.0;
        w_control_change_ = 1.0;
        w_terminal_ = 20.0;

        joint_start_ = 0;
        joint_rate_start_ = n_states_ * mpc_horizon_;
    }

    void operator()(ADvector& fg, const ADvector& vars)
    {
        const size_t nConstraints = n_states_ + (mpc_horizon_ - 1) * n_states_;
        const size_t expected_fg_size = 1 + nConstraints;
        const size_t expected_vars_size = mpc_horizon_ * n_states_ + (mpc_horizon_ - 1) * n_controls_;

        if (fg.size() != expected_fg_size) fg.resize(expected_fg_size);
        assert(vars.size() == expected_vars_size);

        // Cost
        fg[0] = 0;

        // State tracking cost
        for (int t = 0; t < mpc_horizon_ - 1 ; ++t) {
            for (int j = 0; j < n_states_; ++j) {
                size_t idx = joint_start_ + t * n_states_ + j;
                fg[0] += w_joints_[j] * CppAD::pow(vars[idx] - ref_joints_[j], 2);
            }
        }

        // Control effort cost
        for (int t = 0; t < mpc_horizon_ - 1; ++t) {
            for (int j = 0; j < n_controls_; ++j) {
                size_t idx = joint_rate_start_ + t * n_controls_ + j;
                fg[0] += w_control_ * CppAD::pow(vars[idx], 2);
            }
        }

        // Control change cost (smoothness)
        for (int t = 0; t < mpc_horizon_ - 2; ++t) {
            for (int j = 0; j < n_controls_; ++j) {
                size_t idx0 = joint_rate_start_ + t * n_controls_ + j;
                size_t idx1 = joint_rate_start_ + (t + 1) * n_controls_ + j;
                fg[0] += w_control_change_ * CppAD::pow(vars[idx1] - vars[idx0], 2);
            }
        }

        // Terminal cost
        for (int j = 0; j < n_states_; ++j) {
            size_t idx = joint_start_ + (mpc_horizon_ - 1) * n_states_ + j;
            fg[0] += w_terminal_ * CppAD::pow(vars[idx] - ref_joints_[j], 2);
        }

        // Initial state constraints: x0 - initial state = 0
        for (int j = 0; j < n_states_; ++j) {
            fg[1 + j] = vars[joint_start_ + j];
        }

        // Dynamics constraints: x_{t+1} - (x_t + u_t*dt) = 0
        const size_t offset_from_initial = 1 + n_states_;
        for (int t = 0; t < mpc_horizon_ - 1; ++t) {
            for (int j = 0; j < n_states_; ++j) {
                size_t idx_fg = offset_from_initial + t * n_states_ + j;
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
 *
 */

MPC::MPC()
{
    mpc_horizon_ = 20;
    max_rate_ = 5.0;
    bound_value_ = 1.0e3;

    references_.resize(9, 0.0);

    last_controls_.assign(9, 0.0);
}

void MPC::set_references(double j0, double j1, double j2, double j3, double j4, double j5, double j6, double j7, double j8)
{
    references_ = {j0, j1, j2, j3, j4, j5, j6, j7, j8};
}

std::tuple<std::vector<std::vector<double>>, std::vector<double>> MPC::solve(const Eigen::VectorXd& state)
{
    typedef CPPAD_TESTVECTOR(double) Dvector;
    const size_t nStates = 9;
    const size_t nControls = 9;
    const int H = mpc_horizon_;

    // number of decision variables
    const size_t nVars = (size_t)H * nStates + (size_t)(H - 1) * nControls;

    // number of constraints
    const size_t nConstraints = nStates + (size_t)(H - 1) * nStates;

    Dvector vars(nVars);
    Dvector vars_lowerbound(nVars);
    Dvector vars_upperbound(nVars);
    Dvector constraints_lowerbound(nConstraints);
    Dvector constraints_upperbound(nConstraints);

    // --- Initialize decision variables (initial guess) ---
    for (int t = 0; t < H; ++t) {
        double alpha = t / double(H - 1);
        for (size_t j = 0; j < nStates; ++j) {
            vars[t * nStates + j] = (1.0 - alpha) * state[j] + alpha * references_[j];
    }
}

    // --- Variable bounds ---
    for (size_t i = 0; i < nVars; i++)
    {
        if (i < (size_t)nStates * (size_t)H)
        {
            vars_lowerbound[i] = -bound_value_;
            vars_upperbound[i] = bound_value_;
        }
        else
        {
            vars_lowerbound[i] = -max_rate_;
            vars_upperbound[i] = max_rate_;
        }
    }

    // --- Constraints initialization (all zeros by default) ---
    for (size_t i = 0; i < nConstraints; i++)
    {
        constraints_lowerbound[i] = 0.0;
        constraints_upperbound[i] = 0.0;
    }

    // --- Initial state constraints (positions) ---
    for (size_t j = 0; j < nStates; j++)
    {
        constraints_lowerbound[j] = state[j];
        constraints_upperbound[j] = state[j];
        vars[j] = state[j]; // warm-start x0
    }
    // --- Warm-start: initialize the entire control horizon with last_controls_ ---
    const size_t control_block_start = (size_t)nStates * (size_t)H;
    const size_t control_steps = (size_t)(H > 0 ? (H - 1) : 0);
    for (size_t t = 0; t < control_steps; ++t)
    {
        for (size_t j = 0; j < nControls; ++j)
        {
            size_t idx = control_block_start + t * nControls + j;
            if (idx < vars.size() && j < last_controls_.size()) vars[idx] = last_controls_[j];
        }
    }

    // --- Solve ---
    std::vector<double> state_vec(state.data(), state.data() + state.size());
    FG_eval fg_eval(references_, H, 0.1);

    CppAD::ipopt::solve_result<Dvector> solution;

    // Correct IPOPT options (each line must start with a token)
    std::string printOpt;
    printOpt += "Integer print_level  5\n"; // Uncomment for more print information
    printOpt += "Sparse  true        forward\n";
    printOpt += "Sparse  true        reverse\n";
    printOpt +=  "String linear_solver mumps\n";
    printOpt += "Numeric max_cpu_time   0.5\n";
    // --- Solve ---
    CppAD::ipopt::solve<Dvector, FG_eval>(
        printOpt, vars, vars_lowerbound, vars_upperbound,
        constraints_lowerbound, constraints_upperbound,
        fg_eval, solution);

    // Check the results
    bool ok = solution.status == CppAD::ipopt::solve_result<Dvector>::success;
    if (!ok) {
        std::cerr << "MPC solve failed!" << std::endl;
    }

    // --- SAFETY GUARD: ensure solution.x has expected size before indexing it ---
    if (solution.x.size() < nVars) {
        std::cerr << "[MPC] WARNING: solution.x size (" << solution.x.size()
                  << ") < expected nVars (" << nVars << "). Returning safe fallback.\n";

        // Build fallback trajectory: repeat current state across horizon
        std::vector<std::vector<double>> traj;
        traj.reserve(H);
        for (int t = 0; t < H; ++t) {
            std::vector<double> joints(nStates);
            for (size_t j = 0; j < nStates; ++j) {
                joints[j] = static_cast<double>(state[j]); // copy current state
            }
            traj.push_back(joints);
        }

        // Use last known controls as fallback (or zeros if empty)
        std::vector<double> controls(nControls, 0.0);
        if (last_controls_.size() == nControls) controls = last_controls_;

        // Do not update last_controls_ (or you may choose to leave it unchanged)
        return {traj, controls};
    }

    // --- Trajectory (all joints) ---
    std::vector<std::vector<double>> traj;
    traj.reserve(H);
    for (int t = 0; t < H; t++)
    {
        std::vector<double> joints(nStates);
        for (size_t j = 0; j < nStates; j++)
            joints[j] = solution.x[(size_t)t * nStates + j];
        traj.push_back(joints);
    }

    // --- Controls: first timestep ---
    std::vector<double> controls(nControls, 0.0);
    const size_t first_control_index = (size_t)nStates * (size_t)H;
    if (first_control_index + nControls <= solution.x.size())
    {
        for (size_t j = 0; j < nControls; j++)
            controls[j] = solution.x[first_control_index + j];
    }

    // --- Save first controls for next warm-start ---
    last_controls_ = controls;

    return {traj, controls};

}


} // namespace MpcRos
