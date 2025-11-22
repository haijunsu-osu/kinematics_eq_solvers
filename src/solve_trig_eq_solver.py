"""
Analytical solver for a*cos(x) + b*sin(x) + c = 0 using the Weierstrass substitution.

The substitution t = tan(x / 2) rewrites cos(x) and sin(x) as rational expressions
in t, turning the trigonometric equation into a quadratic:

    cos(x) = (1 - t^2) / (1 + t^2)
    sin(x) = 2t / (1 + t^2)

Substituting and clearing the denominator yields

    (c - a) * t^2 + 2b * t + (a + c) = 0.

Solving this polynomial recovers all finite t roots, and every finite t maps back to
an angle through x = 2 * atan(t). The mapping misses the endpoint x = π (equivalently
-π), which corresponds to t → ±∞, so the solver explicitly rechecks that angle.

The solver below carefully handles degenerate cases:
  * Purely constant equations (no sine/cosine terms).
  * Purely sinusoidal or purely cosine equations.
  * Near-singular quadratics where coefficients collapse.
  * Numerical noise for tiny coefficients.
Every reported angle is normalized to [-π, π], deduplicated, and verified against the
original equation within a strict tolerance.  The helper ``stress_test_solver`` runs
a lightweight Monte-Carlo validation augmented with explicit degenerate and single-
solution cases, and it reports a concise summary of the test campaign.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple


_TWO_PI = 2.0 * math.pi
_ANGLE_TOL = 1e-12
_EQ_TOL = 1e-12


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to [-π, π] and snap both endpoints to π for consistency."""
    wrapped = math.remainder(angle, _TWO_PI)
    if wrapped <= -math.pi:
        wrapped += _TWO_PI
    elif wrapped > math.pi:
        wrapped -= _TWO_PI
    if math.isclose(wrapped, -math.pi, abs_tol=_ANGLE_TOL) or math.isclose(
        wrapped, math.pi, abs_tol=_ANGLE_TOL
    ):
        return math.pi
    return wrapped


def _angular_distance(a: float, b: float) -> float:
    """Smallest signed distance from angle b to angle a."""
    diff = a - b
    diff = math.remainder(diff, _TWO_PI)
    if diff <= -math.pi:
        diff += _TWO_PI
    elif diff > math.pi:
        diff -= _TWO_PI
    return diff


def _append_unique(angles: List[float], candidate: float, tol: float) -> None:
    """Append candidate when it is not already captured within tol."""
    for existing in angles:
        if abs(_angular_distance(existing, candidate)) <= tol:
            return
    angles.append(candidate)


def _solve_quadratic(a: float, b: float, c: float, eps: float) -> Sequence[float]:
    """Solve a* t^2 + b* t + c = 0 with safeguards for degenerate coefficients."""
    if abs(a) <= eps:
        if abs(b) <= eps:
            return []  # Either no equation (handled upstream) or inconsistent.
        return [(-c) / b]

    discriminant = b * b - 4.0 * a * c
    if discriminant < -eps:
        return []
    disc = 0.0 if abs(discriminant) <= eps else discriminant
    sqrt_disc = math.sqrt(disc)
    # Use numerically stable quadratic formula.
    if b >= 0:
        q = -0.5 * (b + sqrt_disc)
    else:
        q = -0.5 * (b - sqrt_disc)
    root1 = q / a
    roots = [root1]
    if abs(q) > eps:
        root2 = c / q
        roots.append(root2)
    else:
        roots.append((-b - q) / (2.0 * a))
    return roots


def solve_trig_eq(a: float, b: float, c: float) -> Tuple[List[float], bool]:
    """
    Solve a*cos(x) + b*sin(x) + c = 0 analytically.

    Parameters
    ----------
    a, b, c : float
        Real-valued coefficients.

    Returns
    -------
    solutions : List[float]
        Sorted list of up to two angles in [-π, π] satisfying the equation.
    arbitrary : bool
        True when the equation collapses to 0 = 0 (every x is a solution).

    Notes
    -----
    The general case leverages the Weierstrass substitution t = tan(x/2) to build a
    quadratic in t.  We solve and back-substitute via x = 2 atan(t).  The special
    cases (constant/sine/cosine-only) are handled explicitly for clarity and
    stability.  Each candidate is verified against the original equation with a
    scale-aware tolerance to avoid reporting spurious solutions.
    """
    a = float(a)
    b = float(b)
    c = float(c)
    scale = max(abs(a), abs(b), abs(c), 1.0)
    eq_tol = _EQ_TOL * scale + 1e-15
    tiny = 1e-14 * scale + 1e-15

    # No trigonometric terms present.
    if abs(a) <= tiny and abs(b) <= tiny:
        if abs(c) <= eq_tol:
            return ([], True)
        return ([], False)

    solutions: List[float] = []

    def _verify_and_add(angle: float) -> None:
        norm = _normalize_angle(angle)
        residual = a * math.cos(norm) + b * math.sin(norm) + c
        if abs(residual) <= eq_tol:
            _append_unique(solutions, norm, tol=1e-10)

    # Pure cosine equation.
    if abs(b) <= tiny:
        if abs(a) <= tiny:
            # Already handled earlier, but keep guard.
            return ([], abs(c) <= eq_tol)
        rhs = -c / a
        if abs(rhs) <= 1.0 + 1e-12:
            rhs = max(-1.0, min(1.0, rhs))
            angle = math.acos(rhs)
            _verify_and_add(angle)
            _verify_and_add(-angle)
        return (sorted(solutions), False)

    # Pure sine equation.
    if abs(a) <= tiny:
        rhs = -c / b
        if abs(rhs) <= 1.0 + 1e-12:
            rhs = max(-1.0, min(1.0, rhs))
            angle = math.asin(rhs)
            _verify_and_add(angle)
            _verify_and_add(math.pi - angle)
        return (sorted(solutions), False)

    # General case via the Weierstrass substitution.
    quad_a = c - a
    quad_b = 2.0 * b
    quad_c = a + c
    roots = _solve_quadratic(quad_a, quad_b, quad_c, eps=tiny)
    for t in roots:
        angle = 2.0 * math.atan(t)
        _verify_and_add(angle)

    # Recover potential endpoint root at x = π (t → ±∞).
    _verify_and_add(math.pi)

    solutions.sort()
    return (solutions, False)


def stress_test_solver(
    num_trials: int = 1000,
    rng_seed: int = 73421,
    angle_tol: float = 1e-9,
) -> None:
    """
    Randomized validation that the solver recovers known solutions.

    Each trial constructs coefficients from known angles such that the solver must
    recover those roots.  Additional deterministic edge cases cover degenerate,
    single-solution, and contradictory equations.  The routine prints a concise
    summary once all cases pass.
    """
    rng = random.Random(rng_seed)

    def _contains(container: Iterable[float], angle: float) -> bool:
        return any(abs(_angular_distance(val, angle)) <= angle_tol for val in container)

    # Deterministic degenerate checks.
    deterministic_cases = 0

    sols, arbitrary = solve_trig_eq(0.0, 0.0, 0.0)
    assert arbitrary and not sols
    deterministic_cases += 1

    sols, arbitrary = solve_trig_eq(0.0, 0.0, 1.0)
    assert not arbitrary and not sols
    deterministic_cases += 1

    sols, _ = solve_trig_eq(1.0, 0.0, 1.0)
    assert len(sols) == 1 and math.isclose(sols[0], math.pi, abs_tol=angle_tol)
    deterministic_cases += 1

    sols, _ = solve_trig_eq(1.0, 0.0, 0.0)
    assert len(sols) == 2 and _contains(sols, math.pi / 2) and _contains(sols, -math.pi / 2)
    deterministic_cases += 1

    sols, _ = solve_trig_eq(0.0, 1.0, 0.0)
    assert len(sols) == 2 and _contains(sols, 0.0) and _contains(sols, math.pi)
    deterministic_cases += 1

    sols, _ = solve_trig_eq(0.0, 1.0, 1.0)
    assert len(sols) == 1 and _contains(sols, -math.pi / 2)
    deterministic_cases += 1

    for idx in range(num_trials):
        mode = idx % 6
        if mode == 0:
            a = rng.uniform(-5.0, 5.0)
            b = rng.uniform(-5.0, 5.0)
            if abs(a) < 1e-6 and abs(b) < 1e-6:
                a = 1.0
            true_angle = rng.uniform(-math.pi, math.pi)
            c = -a * math.cos(true_angle) - b * math.sin(true_angle)
            expected = [_normalize_angle(true_angle)]
        elif mode == 1:
            angle = rng.uniform(-math.pi, math.pi)
            a = rng.uniform(0.1, 3.0)
            b = 0.0
            c = -a * math.cos(angle)
            rhs = -c / a
            rhs = max(-1.0, min(1.0, rhs))
            base = math.acos(rhs)
            expected = sorted(
                {
                    _normalize_angle(base),
                    _normalize_angle(-base),
                }
            )
        elif mode == 2:
            angle = rng.uniform(-math.pi, math.pi)
            b = rng.uniform(0.1, 3.0)
            a = 0.0
            c = -b * math.sin(angle)
            expected = sorted(
                {
                    _normalize_angle(angle),
                    _normalize_angle(math.pi - angle),
                }
            )
        elif mode == 3:
            # Force the endpoint solution x = π (c = a makes the numerator vanish).
            a = rng.uniform(-5.0, 5.0)
            b = rng.uniform(-5.0, 5.0)
            c = a
            expected = [_normalize_angle(math.pi)]
        elif mode == 4:
            # Pure constant contradiction.
            a = 0.0
            b = 0.0
            c = rng.uniform(0.1, 5.0)
            expected = []
        else:
            # Random valid quadratic with two distinct roots.
            t1 = rng.uniform(-5.0, 5.0)
            t2 = rng.uniform(-5.0, 5.0)
            # Build coefficients backwards from t roots.
            # Inverse mapping: (c - a) = k, 2b = -k*(t1 + t2), (a + c) = k * t1 * t2
            k = rng.uniform(0.5, 2.0)
            quad_a = k
            quad_b = -k * (t1 + t2)
            quad_c = k * t1 * t2
            # Map back to original coefficients:
            a = 0.5 * (quad_c - quad_a)
            c = 0.5 * (quad_c + quad_a)
            b = 0.5 * quad_b
            expected = sorted(
                {
                    _normalize_angle(2.0 * math.atan(t1)),
                    _normalize_angle(2.0 * math.atan(t2)),
                }
            )

        sols, arbitrary = solve_trig_eq(a, b, c)
        if expected:
            assert not arbitrary
            for tgt in expected:
                assert _contains(sols, tgt), (
                    f"Missing root {tgt} for coefficients {(a, b, c)}; "
                    f"found {sols} (mode {mode})"
                )
        else:
            if abs(a) <= 1e-12 and abs(b) <= 1e-12 and abs(c) <= 1e-12:
                assert arbitrary
            else:
                assert not sols

    total_cases = num_trials + deterministic_cases
    print(
        f"Stress test passed: {total_cases} cases "
        f"({num_trials} randomized + {deterministic_cases} deterministic edge cases) "
        f"with seed {rng_seed}."
    )


if __name__ == "__main__":
    stress_test_solver()
