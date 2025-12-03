/*
 * MIT License
 * 
 * Copyright (c) 2024 Mustafa Ege Kural
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
*/

#ifndef MPC_HPP
#define MPC_HPP

#include <iostream>
#include <map>
#include <math.h>
#include <vector>
#include <tuple>
#include <Eigen/Core>
#include <Eigen/QR>
#include <cppad/ipopt/solve.hpp>

namespace MpcRos
{
class MPC
{
  public:
    MPC();
    MPC(const std::map<std::string, double> &params);
    std::tuple<std::vector<std::vector<double>>, std::vector<double>> solve(const Eigen::VectorXd& state);
    void set_references(double j0, double j1, double j2, double j3, double j4,
                        double j5, double j6, double j7, double j8);
  private:
    int mpc_horizon_;
    double max_rate_;   
    double bound_value_;
    std::vector<double> _references; 
};
} // namespace MpcRos
#endif