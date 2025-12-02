/*
 * MIT License
 * Adapted for robot-arm-pushed vacuum with head joints
 * Author: Mustafa Ege Kural (original)
 * Adaptation: Gabriela Vitez
 */

#pragma once

#include <vector>
#include <tuple>
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <Eigen/Dense>

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
    double dt_;
    int mpc_horizon_;
    std::vector<double> ref_joints_;      // reference for 9 joints
    std::vector<double> w_joints_;        // weights for 9 joints
    double w_control_;
    double w_control_change_;

    int joint_start_;
    int joint_rate_start_;

    typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;

    FG_eval()
    {
        dt_ = 0.1;
        mpc_horizon_ = 20;

        ref_joints_.resize(9, 0.0);
        w_joints_.resize(9, 100.0);

        w_control_ = 1.0;
        w_control_change_ = 1.0;

        joint_start_ = 0;
        joint_rate_start_ = mpc_horizon_;
    }

    void set_references(const std::vector<double>& refs)
    {
        ref_joints_ = refs;
    }

    void operator()(ADvector& fg, const ADvector& vars)
    {
        // fg[0] is cost
        fg[0] = 0;

        // --- State tracking cost ---
        for (int i = 0; i < mpc_horizon_; i++)
        {
            for (size_t j = 0; j < 9; j++)
            {
                fg[0] += w_joints_[j] * CppAD::pow(vars[joint_start_ + i * 9 + j] - ref_joints_[j], 2);
            }
        }

        // --- Control effort cost ---
        for (int i = 0; i < mpc_horizon_ - 1; i++)
        {
            for (size_t j = 0; j < 9; j++)
            {
                fg[0] += w_control_ * CppAD::pow(vars[joint_rate_start_ + i * 9 + j], 2);
            }
        }

        // --- Control change cost ---
        for (int i = 0; i < mpc_horizon_ - 2; i++)
        {
            for (size_t j = 0; j < 9; j++)
            {
                fg[0] += w_control_change_ * CppAD::pow(vars[joint_rate_start_ + (i+1) * 9 + j] - vars[joint_rate_start_ + i * 9 + j], 2);
            }
        }

        // --- Initial constraints (initial state equals first vars) ---
        for (size_t j = 0; j < 9; j++)
        {
            fg[1 + j] = vars[j]; // first timestep
        }

        // --- Dynamics constraints---
        for (int t = 0; t < mpc_horizon_ - 1; t++)
        {
            for (size_t j = 0; j < 9; j++)
            {
                CppAD::AD<double> joint0 = vars[joint_start_ + t * 9 + j];
                CppAD::AD<double> joint1 = vars[joint_start_ + (t + 1) * 9 + j];
                CppAD::AD<double> joint_rate = vars[joint_rate_start_ + t * 9 + j];

                fg[1 + 9 + t * 9 + j] = joint1 - (joint0 + joint_rate * dt_);
            }
        }
    }
};

/**
 * @brief MPC solver class for 9 joints
 */
class MPC
{
public:
    int mpc_horizon_;
    double max_rate_;   
    double bound_value_;

    std::vector<double> _references; 

    MPC()
    {
        mpc_horizon_ = 20;
        max_rate_ = 1.0;
        bound_value_ = 1.0e3;

        _references.resize(9, 0.0);
    }

    void set_references(double j0, double j1, double j2, double j3, double j4, double j5, double j6, double j7, double j8)
    {
        _references = {j0, j1, j2, j3, j4, j5, j6, j7, j8};
    }

    std::tuple<std::vector<std::vector<double>>, std::vector<double>>
    solve(const Eigen::VectorXd& state)
    {
        typedef CPPAD_TESTVECTOR(double) Dvector;
        const size_t nStates = 9;
        const size_t nControls = 9;
        const size_t nVars = mpc_horizon_ * nStates + (mpc_horizon_ - 1) * nControls;
        const size_t nConstraints = nStates + (mpc_horizon_ - 1) * nStates;

        Dvector vars(nVars);
        Dvector vars_lowerbound(nVars);
        Dvector vars_upperbound(nVars);
        Dvector constraints_lowerbound(nConstraints);
        Dvector constraints_upperbound(nConstraints);

        // --- Initialize decision variables ---
        for (size_t i = 0; i < nVars; i++) vars[i] = 0.0;

        // --- Variable bounds ---
        for (size_t i = 0; i < nVars; i++)
        {
            if (i < nStates * mpc_horizon_)
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

        // --- Constraints ---
        for (size_t i = 0; i < nConstraints; i++)
        {
            constraints_lowerbound[i] = 0.0;
            constraints_upperbound[i] = 0.0;
        }
        // Initial state constraints
        for (size_t j = 0; j < 9; j++)
        {
            constraints_lowerbound[j] = state[j];
            constraints_upperbound[j] = state[j];
            vars[j] = state[j];
        }

        // --- Solve ---
        FG_eval fg_eval;
        fg_eval.set_references(_references);
        CppAD::ipopt::solve_result<Dvector> solution;
        std::string options = "Integer print_level 0\nSparse true forward\nSparse true reverse\nNumeric max_cpu_time 0.5\n";

        CppAD::ipopt::solve<Dvector, FG_eval>(
            options, vars, vars_lowerbound, vars_upperbound,
            constraints_lowerbound, constraints_upperbound,
            fg_eval, solution);

        // --- Trajectory (all joints) ---
        std::vector<std::vector<double>> traj;
        for (int t = 0; t < mpc_horizon_; t++)
        {
            std::vector<double> joints(9);
            for (size_t j = 0; j < 9; j++)
                joints[j] = solution.x[t * 9 + j];
            traj.push_back(joints);
        }

        // --- Controls: first timestep ---
        std::vector<double> controls(9);
        for (size_t j = 0; j < 9; j++)
            controls[j] = solution.x[nStates * mpc_horizon_ + j];

        return {traj, controls};
    }
};

} // namespace MpcRos
