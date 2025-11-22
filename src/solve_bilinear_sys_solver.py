"""
Solver for bilinear trigonometric systems using Weierstrass substitution.

We solve K @ m(theta1, theta2) = 0 where K is 2x9 and
    m = [1,
         cos(theta1), sin(theta1),
         cos(theta2), sin(theta2),
         cos(theta1)cos(theta2), cos(theta1)sin(theta2),
         sin(theta1)cos(theta2), sin(theta1)sin(theta2)]^T.

By substituting t1 = tan(theta1/2) and t2 = tan(theta2/2), every trigonometric
entry becomes a rational function whose denominators differ only by
(1 + t1^2) and (1 + t2^2). Multiplying each row equation of K by the common
denominator yields bivariate polynomials p1(t1, t2) and p2(t1, t2), each with
degree at most two in t1 and t2. Eliminating t2 with the Sylvester resultant
leads to a quartic matrix polynomial M(t1) = M0 + M1 t1 + M2 t1^2. The roots of
det M(t1) correspond to feasible t1 values.

Roots are found either via the companion linearization (when M2 is well
conditioned) or through a generalized eigenvalue formulation (when M2 is
singular/ill-conditioned). Each t1 is converted back to theta1, substituted
into the original linear equations, and theta2 is recovered with the already
implemented solve_trig_sys_single() helper. All candidate pairs are then
validated against the original K @ m = 0 system.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from solve_trig_sys_single_solver import solve_trig_sys_single


NUM_TOL = 1e-10
EIG_TOL = 1e-8
ANGLE_TOL = 1e-8
RES_TOL = 1e-8


def solve_bilinear_sys(K: np.ndarray) -> List[Tuple[float, float]]:
    """
    Solve K @ m(theta1, theta2) = 0 using the Weierstrass substitution and
    Sylvester resultant elimination.

    Args:
        K: 2x9 coefficient matrix.

    Returns:
        List of (theta1, theta2) tuples with angles normalized to [-pi, pi].
    """

    K = _validate_matrix(K)
    polys = [_build_polynomial(K[i]) for i in range(2)]
    M0, M1, M2 = _construct_sylvester(polys[0], polys[1])
    t1_list = _solve_t1_roots(M0, M1, M2)

    solutions: List[Tuple[float, float]] = []
    seen = set()

    for t1 in t1_list:
        theta1 = _theta_from_tan_half(t1)
        if theta1 is None:
            continue
        cos1 = math.cos(theta1)
        sin1 = math.sin(theta1)
        A2, c2 = _build_theta2_system(K, cos1, sin1)
        res = solve_trig_sys_single(A2, c2)
        for theta2 in res["solutions"]:
            theta2 = _normalize_angle(theta2)
            if _verify_solution(K, theta1, theta2) and not _solution_seen(
                theta1, theta2, seen
            ):
                solutions.append((theta1, theta2))
                seen.add(
                    (round(theta1 / ANGLE_TOL), round(theta2 / ANGLE_TOL))
                )

    solutions.sort()
    return solutions


def _validate_matrix(K: np.ndarray) -> np.ndarray:
    arr = np.asarray(K, dtype=float)
    if arr.shape != (2, 9):
        raise ValueError("K must have shape (2, 9).")
    return arr


def _build_polynomial(row: np.ndarray) -> np.ndarray:
    """Return coefficient matrix c[i, j] for t1^i * t2^j."""

    coeffs = np.zeros((3, 3), dtype=float)
    for idx, value in enumerate(row):
        coeffs += value * POLY_TERMS[idx]
    return coeffs


def _construct_sylvester(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct M0, M1, M2 for the Sylvester matrix polynomial."""

    a_coeffs = [p1[:, j] for j in range(3)]  # y^j coefficients
    b_coeffs = [p2[:, j] for j in range(3)]
    zero_poly = np.zeros(3)

    rows = [
        [a_coeffs[2], a_coeffs[1], a_coeffs[0], zero_poly],
        [zero_poly, a_coeffs[2], a_coeffs[1], a_coeffs[0]],
        [b_coeffs[2], b_coeffs[1], b_coeffs[0], zero_poly],
        [zero_poly, b_coeffs[2], b_coeffs[1], b_coeffs[0]],
    ]

    M0 = np.zeros((4, 4))
    M1 = np.zeros((4, 4))
    M2 = np.zeros((4, 4))

    for r in range(4):
        for c in range(4):
            poly = rows[r][c]
            for power, value in enumerate(poly):
                if power == 0:
                    M0[r, c] = value
                elif power == 1:
                    M1[r, c] = value
                elif power == 2:
                    M2[r, c] = value
    return M0, M1, M2


def _solve_t1_roots(M0: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> List[float]:
    """Solve det(M0 + M1 t + M2 t^2) = 0 for real t."""

    candidates: List[float] = []
    cond = _matrix_condition(M2)
    if cond < 1e8:
        try:
            inv_M2 = np.linalg.inv(M2)
            top = np.hstack((-inv_M2 @ M1, -inv_M2 @ M0))
            bottom = np.hstack((np.eye(4), np.zeros((4, 4))))
            companion = np.vstack((top, bottom))
            eigvals = np.linalg.eigvals(companion)
            candidates.extend(_filter_real_eigenvalues(eigvals))
        except np.linalg.LinAlgError:
            cond = np.inf

    if cond >= 1e8:
        A = np.block([[-M1, -M0], [np.eye(4), np.zeros((4, 4))]])
        B = np.block([[M2, np.zeros((4, 4))], [np.zeros((4, 4)), np.eye(4)]])
        eigvals = _generalized_eigenvalues(A, B)
        candidates.extend(_filter_real_eigenvalues(eigvals))
        if _is_singular(B):
            candidates.append(np.inf)

    # Always include infinite t1 candidate to capture theta1 = pi
    candidates.append(np.inf)
    filtered: List[float] = []
    for val in candidates:
        if math.isinf(val):
            filtered.append(np.inf)
            continue
        if not any(abs(val - existing) <= 1e-8 for existing in filtered if not math.isinf(existing)):
            filtered.append(val)
    return filtered


def _matrix_condition(M: np.ndarray) -> float:
    try:
        return np.linalg.cond(M)
    except np.linalg.LinAlgError:
        return np.inf


def _filter_real_eigenvalues(values: np.ndarray) -> List[float]:
    roots = []
    for val in values:
        if abs(val.imag) <= EIG_TOL:
            roots.append(float(val.real))
    return roots


def _generalized_eigenvalues(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        sol = np.linalg.solve(B, A)
        return np.linalg.eigvals(sol)
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(B)
        return np.linalg.eigvals(pinv @ A)


def _is_singular(B: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(B.T @ B)
        return False
    except np.linalg.LinAlgError:
        return True


def _theta_from_tan_half(t: float) -> float | None:
    if math.isinf(t):
        return math.pi
    denom = 1.0 + t * t
    if denom <= NUM_TOL:
        return None
    return _normalize_angle(2.0 * math.atan(t))


def _build_theta2_system(K: np.ndarray, cos1: float, sin1: float) -> Tuple[np.ndarray, np.ndarray]:
    A = np.zeros((2, 2))
    c = np.zeros(2)
    for i in range(2):
        row = K[i]
        coeff_cos2 = row[3] + row[5] * cos1 + row[7] * sin1
        coeff_sin2 = row[4] + row[6] * cos1 + row[8] * sin1
        rhs = -(row[0] + row[1] * cos1 + row[2] * sin1)
        A[i, 0] = coeff_cos2
        A[i, 1] = coeff_sin2
        c[i] = rhs
    return A, c


def _verify_solution(K: np.ndarray, theta1: float, theta2: float) -> bool:
    vec = _monomial_vector(theta1, theta2)
    residual = K @ vec
    return float(np.linalg.norm(residual)) <= RES_TOL


def _monomial_vector(theta1: float, theta2: float) -> np.ndarray:
    c1, s1 = math.cos(theta1), math.sin(theta1)
    c2, s2 = math.cos(theta2), math.sin(theta2)
    return np.array(
        [
            1.0,
            c1,
            s1,
            c2,
            s2,
            c1 * c2,
            c1 * s2,
            s1 * c2,
            s1 * s2,
        ]
    )


def _normalize_angle(angle: float) -> float:
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped < 0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def _solution_seen(theta1: float, theta2: float, seen: set) -> bool:
    key = (round(theta1 / ANGLE_TOL), round(theta2 / ANGLE_TOL))
    return key in seen


# Polynomial templates for each monomial entry of m, already multiplied by
# (1 + t1^2)(1 + t2^2).
POLY_TERMS = []
x = "t1"
y = "t2"


def _poly_from_entries(entries: List[Tuple[int, int, float]]) -> np.ndarray:
    poly = np.zeros((3, 3))
    for pow_x, pow_y, coeff in entries:
        if pow_x > 2 or pow_y > 2:
            continue
        poly[pow_x, pow_y] += coeff
    return poly


POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 0, 1.0),
            (2, 0, 1.0),
            (0, 2, 1.0),
            (2, 2, 1.0),
        ]
    )
)  # 1
POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 0, 1.0),
            (0, 2, 1.0),
            (2, 0, -1.0),
            (2, 2, -1.0),
        ]
    )
)  # cos1
POLY_TERMS.append(
    _poly_from_entries(
        [
            (1, 0, 2.0),
            (1, 2, 2.0),
        ]
    )
)  # sin1
POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 0, 1.0),
            (2, 0, 1.0),
            (0, 2, -1.0),
            (2, 2, -1.0),
        ]
    )
)  # cos2
POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 1, 2.0),
            (2, 1, 2.0),
        ]
    )
)  # sin2
POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 0, 1.0),
            (2, 2, 1.0),
            (2, 0, -1.0),
            (0, 2, -1.0),
        ]
    )
)  # cos1 cos2
POLY_TERMS.append(
    _poly_from_entries(
        [
            (0, 1, 2.0),
            (2, 1, -2.0),
        ]
    )
)  # cos1 sin2
POLY_TERMS.append(
    _poly_from_entries(
        [
            (1, 0, 2.0),
            (1, 2, -2.0),
        ]
    )
)  # sin1 cos2
POLY_TERMS.append(
    _poly_from_entries(
        [
            (1, 1, 4.0),
        ]
    )
)  # sin1 sin2


def run_validation_suite(random_trials: int = 1000, seed: int | None = 0) -> int:
    """
    Execute deterministic edge cases followed by randomized solvable systems.
    Logs per-test statistics (root count, residuals, runtime).
    """

    total = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal total
        if not condition:
            raise AssertionError(message)
        print(f"ASSERTION: {message}")
        total += 1

    rng = np.random.default_rng(seed)

    # Deterministic tests -------------------------------------------------------
    theta1, theta2 = 0.6, -1.2
    K = _random_system_from_angles(rng, theta1, theta2)
    sol = _solve_and_report("deterministic-1 (generic)", K)
    expect(_pair_present(sol, theta1, theta2), "Recovered generic angles.")

    # Theta1 = pi test (t1 infinite)
    theta1, theta2 = math.pi, 0.7
    K = _random_system_from_angles(rng, theta1, theta2)
    sol = _solve_and_report("deterministic-2 (theta1=pi)", K)
    expect(_pair_present(sol, theta1, theta2), "Recovered theta1=pi case.")

    # Nearly singular M2 forcing generalized eigenvalue path.
    K = np.zeros((2, 9))
    K[0, 2] = 1.0  # sin1 term
    K[0, 4] = -1.0  # sin2 term
    K[1, 1] = 1.0  # cos1
    K[1, 3] = -1.0  # cos2
    sol = _solve_and_report("deterministic-3 (degenerate M2)", K)
    expect(len(sol) > 0, "Handled degenerate M2 case.")

    # Inconsistent system forcing impossible cosine values.
    inconsistent_K = np.zeros((2, 9))
    inconsistent_K[0, 1] = 1.0  # cos theta1
    inconsistent_K[0, 0] = -2.0  # constant
    inconsistent_K[1, 3] = 1.0  # cos theta2
    inconsistent_K[1, 0] = -2.0
    sol = _solve_and_report("deterministic-4 (inconsistent)", inconsistent_K)
    expect(len(sol) == 0, "Detected inconsistent system.")

    # Randomized testing --------------------------------------------------------
    trials = 0
    while trials < random_trials:
        theta1 = float(rng.uniform(-math.pi, math.pi))
        theta2 = float(rng.uniform(-math.pi, math.pi))
        K = _random_system_from_angles(rng, theta1, theta2)
        label = f"random-{trials + 1}"
        sol = _solve_and_report(label, K)
        if not _pair_present(sol, theta1, theta2):
            raise AssertionError("Randomized validation failed.")
        total += 1
        trials += 1

    return total


def _solve_and_report(label: str, K: np.ndarray) -> List[Tuple[float, float]]:
    start = time.perf_counter()
    sol = solve_bilinear_sys(K)
    elapsed = time.perf_counter() - start
    max_res = 0.0
    for theta1, theta2 in sol:
        vec = _monomial_vector(theta1, theta2)
        max_res = max(max_res, float(np.linalg.norm(K @ vec)))
    if not sol:
        max_res = 0.0
    print(
        f"[{label}] roots={len(sol)} max_res={max_res:.2e} "
        f"time={elapsed * 1e3:.3f} ms"
    )
    if "deterministic" in label or "8" in str(len(sol)):
        print(f"K = {K}")
        print(f"Solutions: {sol}")
    return sol


def _pair_present(
    solutions: Sequence[Tuple[float, float]],
    theta1: float,
    theta2: float,
    tol: float = 1e-6,
) -> bool:
    for t1, t2 in solutions:
        if (
            abs(_normalize_angle(t1 - theta1)) <= tol
            and abs(_normalize_angle(t2 - theta2)) <= tol
        ):
            return True
    return False


def _random_system_from_angles(
    rng: np.random.Generator,
    theta1: float,
    theta2: float,
) -> np.ndarray:
    vec = _monomial_vector(theta1, theta2)
    denom = vec @ vec
    rows = []
    for _ in range(2):
        row = rng.uniform(-1, 1, size=9)
        row -= (row @ vec) / denom * vec
        rows.append(row)
    return np.vstack(rows)


if __name__ == "__main__":
    total = run_validation_suite()
    print(f"Validation succeeded for {total} test systems.")
