"""Coordinate-scaled directions for joint attention-output codeword learning."""
import math
import torch
from gauss_newton import OutputGaussNewton, conjugate_gradient, inner


def group_coordinate_scale(parent_books):
    """Freeze one RMS unit per side/head/group, independent of reconstruction error."""
    assert parent_books.ndim == 5 and parent_books.shape[0] == 2
    unit = parent_books.double().square().mean((-2, -1), keepdim=True).sqrt()
    return unit.clamp_min(1e-6).to(parent_books.dtype).expand_as(parent_books).contiguous()


def propose_direction(residual_functions, point, method, cg_iterations=24, cg_tolerance=1e-4):
    """Residuals already decode physical books from normalized coordinates.

    Both methods use the same relative-coordinate trust radius and subsequent
    FP16/native reassignment acceptance. No learning-rate or damping choice uses
    validation/test performance. A truncated CG result is explicitly identified.
    """
    if method not in ("gradient", "gauss_newton"):
        raise ValueError("unknown joint-output optimizer")
    tiny_damping = 1e-12
    base = OutputGaussNewton(residual_functions, point, tiny_damping)
    gradient, loss = base.gradient_and_loss()
    g2 = float(inner(gradient, gradient))
    if g2 == 0:
        zero = torch.zeros_like(point)
        return zero, gradient, zero, dict(method=method, zero_gradient=True, loss_before=loss)
    if not math.isfinite(g2):
        raise ValueError("invalid output gradient")
    if method == "gradient":
        step = -gradient
        details = dict(method=method, cg=None, damping=None, gradient_rayleigh=None)
    else:
        # Derive the curvature scale from this FIT residual Jacobian, not a
        # raw-K/Fisher surrogate and not the arbitrary synthetic-test damping.
        h0g = base.matvec(gradient) - tiny_damping * gradient
        rayleigh = float(inner(gradient, h0g)) / g2
        if not math.isfinite(rayleigh) or rayleigh < -1e-10:
            raise ValueError("invalid output-Jacobian curvature")
        damping = max(1e-12, .01 * max(0., rayleigh))
        system = OutputGaussNewton(residual_functions, point, damping)
        step, cg = conjugate_gradient(system.matvec, -gradient, cg_iterations, cg_tolerance)
        product = system.matvec(step)
        predicted = -float(inner(gradient, step) + .5 * inner(step, product))
        if predicted <= 0:
            raise ValueError("nondecreasing Gauss-Newton local model")
        details = dict(method=method, cg=cg, damping=damping, gradient_rayleigh=rayleigh,
                       predicted_damped_model_reduction=predicted)
    norm = math.sqrt(float(inner(step, step)))
    radius = math.sqrt(float(inner(point, point)))
    if norm == 0 or radius == 0:
        raise ValueError("degenerate codeword direction")
    direction = step / norm * radius
    slope = float(inner(gradient, direction))
    if not math.isfinite(slope) or slope >= 0:
        raise ValueError("codeword direction is not descent")
    details.update(zero_gradient=False, loss_before=loss, gradient_dot_direction=slope,
                   relative_radius=1., deployed_acceptance_checked=False)
    return direction.detach(), gradient.detach(), step.detach(), details
