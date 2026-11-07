"""Matrix-free Gauss-Newton codeword subproblem for attention residuals.

Residual functions must include the chosen normalization and use fixed hard
assignments. Half projection, new assignments and deployment acceptance belong
outside this continuous subproblem. This module never assigns or deploys codes.
"""
import math
import torch


def inner(a, b):
    return torch.dot(a.reshape(-1).double(), b.reshape(-1).double())


def require_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + " contains a non-finite value")


class OutputGaussNewton:
    """Streaming J^T J products for 1/2 sum_w ||r_w(c)||^2.

    r_w is a projected attention-output residual, including both reconstructed
    K and V when supplied by the caller. No tokenwise or cross-side blocks are
    dropped. Damping penalizes the proposed increment, not raw-K error.
    """

    def __init__(self, residual_functions, point, damping):
        if point.ndim != 1 or not point.is_floating_point():
            raise ValueError("point must be a one-dimensional floating tensor")
        if not math.isfinite(damping) or damping <= 0:
            raise ValueError("strictly positive finite damping is required")
        self.functions = tuple(residual_functions)
        if not self.functions:
            raise ValueError("at least one residual function is required")
        require_finite(point, "point")
        self.point = point.detach().clone()
        self.damping = float(damping)

    def gradient_and_loss(self):
        gradient = torch.zeros_like(self.point)
        loss = self.point.new_zeros((), dtype=torch.float64)
        for function in self.functions:
            residual, pullback = torch.func.vjp(function, self.point)
            require_finite(residual, "residual")
            term, = pullback(residual)
            require_finite(term, "gradient")
            gradient.add_(term.detach())
            loss += 0.5 * inner(residual.detach(), residual.detach())
            del pullback, residual, term
        require_finite(gradient, "summed gradient")
        return gradient, float(loss)

    def matvec(self, direction):
        if direction.shape != self.point.shape or direction.dtype != self.point.dtype:
            raise ValueError("direction shape/dtype must match the point")
        require_finite(direction, "direction")
        answer = self.damping * direction
        for function in self.functions:
            residual, pullback = torch.func.vjp(function, self.point)
            _, tangent = torch.func.jvp(function, (self.point,), (direction,))
            require_finite(residual, "residual")
            require_finite(tangent, "JVP")
            term, = pullback(tangent)
            require_finite(term, "VJP")
            answer = answer + term.detach()
            del pullback, residual, tangent, term
        require_finite(answer, "Gauss-Newton product")
        return answer.detach()


def conjugate_gradient(matvec, rhs, max_iterations=32, relative_tolerance=1e-5):
    """Solve a strictly positive definite system without materializing it.

    A truncated solution is returned explicitly as not converged. Non-finite
    or nonpositive curvature is an error, never an accepted training step.
    """
    if max_iterations < 1 or not 0 < relative_tolerance < 1:
        raise ValueError("invalid CG iteration count or tolerance")
    require_finite(rhs, "right-hand side")
    x = torch.zeros_like(rhs)
    residual = rhs.detach().clone()
    squared = inner(residual, residual)
    initial = float(squared)
    if initial == 0:
        return x, dict(converged=True, iterations=0, relative_residual=0.0)
    direction = residual.clone()
    iterations = 0
    for index in range(max_iterations):
        product = matvec(direction)
        require_finite(product, "matrix-vector product")
        curvature = inner(direction, product)
        if not bool(torch.isfinite(curvature)) or float(curvature) <= 0:
            raise ValueError("CG operator is not positive definite")
        alpha = (squared / curvature).to(rhs.dtype)
        x = x + alpha * direction
        residual = residual - alpha * product
        next_squared = inner(residual, residual)
        iterations = index + 1
        if float(next_squared) <= initial * relative_tolerance ** 2:
            squared = next_squared
            break
        beta = (next_squared / squared).to(rhs.dtype)
        direction = residual + beta * direction
        squared = next_squared
    # Recompute the residual to detect recurrence drift.
    actual = rhs - matvec(x)
    actual_relative = math.sqrt(float(inner(actual, actual)) / initial)
    require_finite(x, "CG solution")
    return x.detach(), dict(
        converged=actual_relative <= relative_tolerance,
        iterations=iterations, relative_residual=actual_relative,
        recurrence_relative_residual=math.sqrt(float(squared) / initial))


def propose(system, max_iterations=32, relative_tolerance=1e-5):
    gradient, loss = system.gradient_and_loss()
    delta, cg = conjugate_gradient(system.matvec, -gradient,
                                   max_iterations, relative_tolerance)
    product = system.matvec(delta)
    linear = float(inner(gradient, delta))
    predicted = -(linear + 0.5 * float(inner(delta, product)))
    if float(inner(gradient, gradient)) > 0 and (linear >= 0 or predicted <= 0):
        raise ValueError("proposal does not decrease the damped local model")
    return delta, dict(
        loss_before=loss, gradient_dot_delta=linear,
        predicted_damped_model_reduction=predicted, cg=cg,
        deployed_acceptance_checked=False, native_assignment_checked=False)
