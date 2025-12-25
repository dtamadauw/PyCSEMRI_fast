#ifndef LSQCPP_WORKSPACE_HPP_
#define LSQCPP_WORKSPACE_HPP_

#include <Eigen/Core>

namespace lsqcpp
{

// --- FIX ---
// Define the Index type alias here, so it is in scope
// for the LsqWorkspace constructor.
// This was previously in lsqcpp.hpp.
using Index = Eigen::MatrixXd::Index;
// --- END FIX ---


/**
 * @brief Pre-allocated workspace for lsqcpp optimizers.
 *
 * This struct holds all the dynamic matrices required by the optimizer
 * during a run. By allocating it once and reusing it, we can
 * avoid heap allocations inside the pixel-by-pixel fitting loop.
 */
template<typename _Scalar,
         int _Inputs = Eigen::Dynamic,
         int _Outputs = Eigen::Dynamic>
struct LsqWorkspace
{
    // --- Type Definitions ---
    using Scalar = _Scalar;
    static constexpr int Inputs = _Inputs;
    static constexpr int Outputs = _Outputs;

    // These types mirror the ones inside LeastSquaresAlgorithm
    using InputVector    = Eigen::Matrix<Scalar, Inputs, 1>;
    using OutputVector   = Eigen::Matrix<Scalar, Outputs, 1>;
    using JacobiMatrix   = Eigen::Matrix<Scalar, Outputs, Inputs>;
    using HessianMatrix  = Eigen::Matrix<Scalar, Inputs, Inputs>;
    using GradientVector = Eigen::Matrix<Scalar, Inputs, 1>;
    using StepVector     = Eigen::Matrix<Scalar, Inputs, 1>;

    // --- Member Variables ---

    // From minimize()
    InputVector    xval;       // Stores the current parameter vector
    OutputVector   fval;       // Stores the residual vector f(x)
    JacobiMatrix   jacobian;   // Stores the Jacobian J(x)
    GradientVector gradient;   // Stores the gradient J^T * f
    StepVector     step;       // Stores the computed step (e.g., LM step)

    // From LevenbergMarquardtMethod()
    HessianMatrix  jacobianSq; // Stores J^T * J
    HessianMatrix  A;          // Stores (J^T * J + lambda*I)
    InputVector    temp_xval;  // Stores x_new for error checking
    OutputVector   temp_fval;  // Stores f(x_new)
    JacobiMatrix   temp_jacobian; // Stores J(x_new)

    // --- Constructor ---

    /**
     * @brief Allocates memory for the workspace (Dynamic version).
     * @param n Number of residuals (Outputs, e.g., nte * 2)
     * @param p Number of parameters (Inputs, e.g., 6)
     */
    LsqWorkspace(Index n, Index p) // This line will now compile
    {
        // Only resize if the type is Dynamic
        if constexpr (Inputs == Eigen::Dynamic)
        {
            xval.resize(p);
            gradient.resize(p);
            step.resize(p);
            jacobianSq.resize(p, p);
            A.resize(p, p);
            temp_xval.resize(p);
        }

        if constexpr (Outputs == Eigen::Dynamic)
        {
            fval.resize(n);
            temp_fval.resize(n);
        }

        if constexpr (Inputs == Eigen::Dynamic || Outputs == Eigen::Dynamic)
        {
            jacobian.resize(n, p);
            temp_jacobian.resize(n, p);
        }
    }

    /**
     * @brief Default constructor for fixed-size (no-op).
     * This allows creation on the stack if dimensions are known at compile time.
     */
    LsqWorkspace()
    {
        static_assert(Inputs != Eigen::Dynamic && Outputs != Eigen::Dynamic,
            "Dynamic workspace must be initialized with (n, p)");
    }
};

// Type alias for your specific dynamic case
using LsqWorkspaceX = LsqWorkspace<double, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace lsqcpp

#endif // LSQCPP_WORKSPACE_HPP_

