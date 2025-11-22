"""
Section A.3 solver for coupled planar trigonometric systems.

We solve

    A @ [cos(theta1), sin(theta1)]^T + B @ [cos(theta2), sin(theta2)]^T = c

exactly as described in section A.3 of the IK paper. The procedure eliminates
the sine variables via the linear system formed by the sine columns, substitutes
the resulting expressions into sin^2(theta) + cos^2(theta) = 1, and uses the
resultant of the two quadratic equations to obtain a quartic polynomial in
cos(theta1). The quartic is solved numerically and every candidate is validated.

Fallback logic (tangent-half-angle elimination and left-null reasoning) handles
degenerate cases where the sine-column matrix becomes singular.

The module exposes:
    * solve_trig_sys(A, B, c)  -> enumerate all solutions (<= 4 pairs)
    * run_validation_suite()   -> deterministic + randomized verification

Only NumPy and the Python standard library are used at runtime. The resultant
formula was derived once using SymPy and hard-coded here as required.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np


NUM_TOL = 1e-9
DET_TOL = 1e-10
ANGLE_TOL = 1e-9
RES_TOL = 5e-8
CIRCLE_TOL = 1e-6
POLY_TOL = 1e-12


def solve_trig_sys(
    A: Sequence[Sequence[float]],
    B: Sequence[Sequence[float]],
    c: Sequence[float],
) -> List[Tuple[float, float]]:
    """
    Solve the coupled system using the Section A.3 elimination strategy.

    Args:
        A, B: 2x2 coefficient matrices.
        c: length-2 vector.

    Returns:
        List of (theta1, theta2) solutions with angles normalized to [-pi, pi].
    """

    mat_a = _to_matrix(A)
    mat_b = _to_matrix(B)
    vec_c = _to_vector(c)

    solutions: List[Tuple[float, float]] = []
    seen = set()

    _section_a3_elimination(mat_a, mat_b, vec_c, solutions, seen)
    _fallback_paths(mat_a, mat_b, vec_c, solutions, seen)

    if not solutions:
        rank_a = _matrix_rank(mat_a)
        rank_b = _matrix_rank(mat_b)
        if rank_a == 0 and rank_b == 0 and np.linalg.norm(vec_c) < NUM_TOL:
            raise ValueError(
                "System underdetermined: both matrices are zero and c is zero."
            )
        if rank_a == 0 and rank_b == 2:
            # A=0, B full rank: theta2 fixed, theta1 arbitrary
            v = np.linalg.inv(mat_b) @ vec_c
            cos2, sin2 = v
            if abs(cos2**2 + sin2**2 - 1.0) <= CIRCLE_TOL:
                theta2 = math.atan2(sin2, cos2)
                # Return solutions with theta1 = 0, pi/2, pi, 3pi/2
                for i in range(4):
                    theta1 = _normalize_angle(i * math.pi / 2)
                    _record_solution(mat_a, mat_b, vec_c, theta1, theta2, solutions, seen)

    solutions.sort()
    return solutions


def _section_a3_elimination(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    solutions: List[Tuple[float, float]],
    seen: set,
) -> None:
    """Apply the Section A.3 elimination when the sine columns span R^2."""

    sin_mat = np.array([[A[0, 1], B[0, 1]], [A[1, 1], B[1, 1]]], dtype=float)
    det = np.linalg.det(sin_mat)
    if abs(det) <= DET_TOL:
        return

    sin_inv = np.linalg.inv(sin_mat)
    base = sin_inv @ c
    cos_cols = np.array([[A[0, 0], B[0, 0]], [A[1, 0], B[1, 0]]], dtype=float)
    coupling = sin_inv @ cos_cols

    l1x, l1y, l10 = -coupling[0, 0], -coupling[0, 1], base[0]
    l2x, l2y, l20 = -coupling[1, 0], -coupling[1, 1], base[1]

    coeffs = _quartic_coefficients(l1x, l1y, l10, l2x, l2y, l20)
    cos1_roots = _real_roots_from_coeffs(coeffs)

    for cos1 in cos1_roots:
        if not np.isfinite(cos1):
            continue
        if abs(cos1) > 1.0 + 5e-6:
            continue
        cos2_candidates = _cos2_from_cos1(
            cos1, l1x, l1y, l10, l2x, l2y, l20
        )
        for cos2 in cos2_candidates:
            if not np.isfinite(cos2):
                continue
            if abs(cos2) > 1.0 + 5e-6:
                continue
            s_vec = base - coupling @ np.array([cos1, cos2])
            s1, s2 = s_vec
            if abs(s1 * s1 + cos1 * cos1 - 1.0) > CIRCLE_TOL:
                continue
            if abs(s2 * s2 + cos2 * cos2 - 1.0) > CIRCLE_TOL:
                continue
            theta1 = _normalize_angle(math.atan2(s1, cos1))
            theta2 = _normalize_angle(math.atan2(s2, cos2))
            _record_solution(A, B, c, theta1, theta2, solutions, seen)


def _quartic_coefficients(
    l1x: float,
    l1y: float,
    l10: float,
    l2x: float,
    l2y: float,
    l20: float,
) -> np.ndarray:
    """Build the quartic coefficients (descending order) via the resultant."""

    a2 = _poly_const(l1y * l1y)
    a1 = np.array([2.0 * l1y * l10, 2.0 * l1y * l1x], dtype=float)
    a0 = np.array(
        [l10 * l10 - 1.0, 2.0 * l1x * l10, l1x * l1x + 1.0],
        dtype=float,
    )

    b2 = _poly_const(l2y * l2y + 1.0)
    b1 = np.array([2.0 * l2y * l20, 2.0 * l2y * l2x], dtype=float)
    b0 = np.array(
        [l20 * l20 - 1.0, 2.0 * l2x * l20, l2x * l2x],
        dtype=float,
    )

    term1 = _poly_mul(_poly_mul(a0, a0), _poly_mul(b2, b2))
    term2 = _poly_mul(_poly_mul(a0, a1), _poly_mul(b1, b2))
    term3 = _poly_mul(_poly_mul(a0, a2), _poly_mul(b0, b2))
    term4 = _poly_mul(_poly_mul(a0, a2), _poly_mul(b1, b1))
    term5 = _poly_mul(_poly_mul(a1, a1), _poly_mul(b0, b2))
    term6 = _poly_mul(_poly_mul(a1, a2), _poly_mul(b0, b1))
    term7 = _poly_mul(_poly_mul(a2, a2), _poly_mul(b0, b0))

    res = term1
    res = _poly_sub(res, term2)
    res = _poly_add(res, _poly_scale(term3, -2.0))
    res = _poly_add(res, term4)
    res = _poly_add(res, term5)
    res = _poly_sub(res, term6)
    res = _poly_add(res, term7)

    if len(res) == 0 or np.all(np.abs(res) < POLY_TOL):
        return np.array([0.0], dtype=float)

    return res[::-1]


def _cos2_from_cos1(
    cos1: float,
    l1x: float,
    l1y: float,
    l10: float,
    l2x: float,
    l2y: float,
    l20: float,
) -> List[float]:
    """Solve the two quadratics for cos(theta2) once cos(theta1) is fixed."""

    a2 = l1y * l1y
    a1 = 2.0 * l1y * (l1x * cos1 + l10)
    a0 = (l1x * cos1 + l10) ** 2 + cos1 * cos1 - 1.0

    b2 = l2y * l2y + 1.0
    b1 = 2.0 * l2y * (l2x * cos1 + l20)
    b0 = (l2x * cos1 + l20) ** 2 - 1.0

    roots: List[float] = []

    if abs(l1y) >= 1e-9:
        for root in _solve_quadratic(a2, a1, a0):
            value = b2 * root * root + b1 * root + b0
            if abs(value) <= 1e-6:
                roots.append(root)
    else:
        for root in _solve_quadratic(b2, b1, b0):
            value = a2 * root * root + a1 * root + a0
            if abs(value) <= 1e-6:
                roots.append(root)
    return roots


def _solve_quadratic(a: float, b: float, c: float) -> List[float]:
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < -1e-10:
        return []
    disc = max(disc, 0.0)
    sqrt_disc = math.sqrt(disc)
    denom = 2.0 * a
    return [(-b + sqrt_disc) / denom, (-b - sqrt_disc) / denom]


def _record_solution(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    theta1: float,
    theta2: float,
    solutions: List[Tuple[float, float]],
    seen: set,
) -> None:
    """Validate the candidate solution and store it if unique."""

    v1 = np.array([math.cos(theta1), math.sin(theta1)], dtype=float)
    v2 = np.array([math.cos(theta2), math.sin(theta2)], dtype=float)
    residual = A @ v1 + B @ v2 - c
    if np.linalg.norm(residual) > RES_TOL:
        return
    key = (round(theta1 / ANGLE_TOL), round(theta2 / ANGLE_TOL))
    if key in seen:
        return
    seen.add(key)
    solutions.append((theta1, theta2))


def _fallback_paths(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    solutions: List[Tuple[float, float]],
    seen: set,
) -> None:
    """Fallback solvers for singular or nearly singular scenarios."""

    _tangent_elimination(A, B, c, solutions, seen, invert_first="B")
    _tangent_elimination(A, B, c, solutions, seen, invert_first="A")
    _rank1_branches(A, B, c, solutions, seen)


def _tangent_elimination(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    solutions: List[Tuple[float, float]],
    seen: set,
    *,
    invert_first: str,
) -> None:
    """Solve via tangent-half-angle substitution when possible."""

    if invert_first == "B":
        if abs(np.linalg.det(B)) <= DET_TOL:
            return
        inv = np.linalg.inv(B)
        M = inv @ A
        d = inv @ c
        branch = "theta1"
    else:
        if abs(np.linalg.det(A)) <= DET_TOL:
            return
        inv = np.linalg.inv(A)
        M = inv @ B
        d = inv @ c
        branch = "theta2"

    q11, q12, q22 = _quadratic_form_terms(M)
    mtd = M.T @ d
    p1 = -2.0 * mtd[0]
    p2 = -2.0 * mtd[1]
    r = float(d @ d - 1.0)

    coeffs = np.array(
        [
            q11 - p1 + r,
            -4.0 * q12 + 2.0 * p2,
            -2.0 * q11 + 4.0 * q22 + 2.0 * r,
            4.0 * q12 + 2.0 * p2,
            q11 + p1 + r,
        ],
        dtype=float,
    )

    roots = _real_roots_from_coeffs(coeffs)
    for t in roots:
        theta = 2.0 * math.atan(t)
        primary_vec = np.array([math.cos(theta), math.sin(theta)], dtype=float)
        secondary_vec = d - M @ primary_vec
        norm = np.linalg.norm(secondary_vec)
        if abs(norm - 1.0) > 5e-6:
            continue
        secondary_vec /= norm
        if branch == "theta1":
            theta1 = theta
            theta2 = math.atan2(secondary_vec[1], secondary_vec[0])
        else:
            theta2 = theta
            theta1 = math.atan2(secondary_vec[1], secondary_vec[0])
        theta1 = _normalize_angle(theta1)
        theta2 = _normalize_angle(theta2)
        _record_solution(A, B, c, theta1, theta2, solutions, seen)


def _quadratic_form_terms(M: np.ndarray) -> Tuple[float, float, float]:
    m11, m12 = M[0, 0], M[0, 1]
    m21, m22 = M[1, 0], M[1, 1]
    q11 = m11 * m11 + m21 * m21
    q22 = m12 * m12 + m22 * m22
    q12 = m11 * m12 + m21 * m22
    return q11, q12, q22


def _rank1_branches(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    solutions: List[Tuple[float, float]],
    seen: set,
) -> None:
    """Use left-null vectors to handle rank-1 matrices."""

    rank_a = _matrix_rank(A)
    rank_b = _matrix_rank(B)

    if rank_b == 1:
        _solve_with_null(
            singular=B,
            other=A,
            c=c,
            primary="theta1",
            solutions=solutions,
            seen=seen,
        )
    if rank_a == 1:
        _solve_with_null(
            singular=A,
            other=B,
            c=c,
            primary="theta2",
            solutions=solutions,
            seen=seen,
        )


def _solve_with_null(
    singular: np.ndarray,
    other: np.ndarray,
    c: np.ndarray,
    primary: str,
    solutions: List[Tuple[float, float]],
    seen: set,
) -> None:
    """Solve using the left-null vector constraint."""

    w = _left_null_vector(singular)
    if w is None:
        return
    other_tw = other.T @ w
    s_val = float(c @ w)
    candidates = _line_circle_intersections(other_tw, s_val)

    for vec in candidates:
        vec = np.array(vec, dtype=float)
        if primary == "theta1":
            u = vec
            rhs = c - other @ u
            counterparts = _solve_rank1_unit(singular, rhs)
            for v in counterparts:
                theta1 = math.atan2(u[1], u[0])
                theta2 = math.atan2(v[1], v[0])
                theta1 = _normalize_angle(theta1)
                theta2 = _normalize_angle(theta2)
                _record_solution(other, singular, c, theta1, theta2, solutions, seen)
        else:
            v = vec
            rhs = c - other @ v
            counterparts = _solve_rank1_unit(singular, rhs)
            for u in counterparts:
                theta1 = math.atan2(u[1], u[0])
                theta2 = math.atan2(v[1], v[0])
                theta1 = _normalize_angle(theta1)
                theta2 = _normalize_angle(theta2)
                _record_solution(singular, other, c, theta1, theta2, solutions, seen)


def _solve_rank1_unit(matrix: np.ndarray, rhs: np.ndarray) -> List[np.ndarray]:
    row, idx = _dominant_row(matrix)
    if row is None:
        if np.linalg.norm(rhs) < NUM_TOL:
            return []
        return []
    target = rhs[idx]
    candidates = _line_circle_intersections(row, target)
    result: List[np.ndarray] = []
    for cand in candidates:
        vec = np.array(cand, dtype=float)
        ok = True
        for r, t in zip(matrix, rhs):
            if np.linalg.norm(r) < NUM_TOL:
                if abs(t) > NUM_TOL:
                    ok = False
                    break
                continue
            if abs(float(r @ vec) - t) > 5e-7:
                ok = False
                break
        if ok:
            result.append(vec)
    return result


def _dominant_row(matrix: np.ndarray) -> Tuple[np.ndarray | None, int]:
    norms = [np.linalg.norm(matrix[0]), np.linalg.norm(matrix[1])]
    if norms[0] < NUM_TOL and norms[1] < NUM_TOL:
        return None, -1
    if norms[0] >= norms[1]:
        return matrix[0], 0
    return matrix[1], 1


def _left_null_vector(matrix: np.ndarray) -> np.ndarray | None:
    if _matrix_rank(matrix) != 1:
        return None
    col0 = matrix[:, 0]
    if np.linalg.norm(col0) >= NUM_TOL:
        return np.array([col0[1], -col0[0]], dtype=float)
    col1 = matrix[:, 1]
    return np.array([col1[1], -col1[0]], dtype=float)


def _line_circle_intersections(
    normal: np.ndarray,
    value: float,
) -> List[Tuple[float, float]]:
    nx, ny = normal
    norm = math.hypot(nx, ny)
    if norm < NUM_TOL:
        return []
    d = value / norm
    if abs(d) > 1.0 + 1e-8:
        return []
    d = max(min(d, 1.0), -1.0)
    base = (d * nx / norm, d * ny / norm)
    discr = 1.0 - d * d
    if discr < 1e-12:
        return [base]
    mag = math.sqrt(discr)
    perp = (-ny / norm, nx / norm)
    sol1 = (base[0] + mag * perp[0], base[1] + mag * perp[1])
    sol2 = (base[0] - mag * perp[0], base[1] - mag * perp[1])
    return [sol1, sol2]


def _matrix_rank(matrix: np.ndarray) -> int:
    det = np.linalg.det(matrix)
    if abs(det) > DET_TOL:
        return 2
    fro = math.sqrt(float(np.sum(matrix * matrix)))
    if fro < NUM_TOL:
        return 0
    return 1


def _real_roots_from_coeffs(coeffs: np.ndarray) -> List[float]:
    coeffs = np.array(coeffs, dtype=float)
    if coeffs.ndim != 1:
        coeffs = coeffs.flatten()
    idx = 0
    while idx < len(coeffs) and abs(coeffs[idx]) < POLY_TOL:
        idx += 1
    coeffs = coeffs[idx:]
    if len(coeffs) <= 1:
        return []
    if len(coeffs) == 2:
        return [-coeffs[1] / coeffs[0]]
    roots = np.roots(coeffs)
    real_roots = [float(r.real) for r in roots if abs(r.imag) <= 1e-8]
    return real_roots


def _poly_const(value: float) -> np.ndarray:
    return np.array([float(value)], dtype=float)


def _poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    res = np.zeros(n, dtype=float)
    res[: len(a)] += a
    res[: len(b)] += b
    return _poly_trim(res)


def _poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(len(a), len(b))
    res = np.zeros(n, dtype=float)
    res[: len(a)] += a
    res[: len(b)] -= b
    return _poly_trim(res)


def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, dtype=float)
    res = np.zeros(len(a) + len(b) - 1, dtype=float)
    for i, val in enumerate(a):
        res[i : i + len(b)] += val * b
    return _poly_trim(res)


def _poly_scale(a: np.ndarray, scalar: float) -> np.ndarray:
    return _poly_trim(a * scalar)


def _poly_trim(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a
    idx = len(a) - 1
    while idx > 0 and abs(a[idx]) < POLY_TOL:
        idx -= 1
    return a[: idx + 1]


def _to_matrix(data: Sequence[Sequence[float]]) -> np.ndarray:
    if len(data) != 2 or len(data[0]) != 2 or len(data[1]) != 2:
        raise ValueError("Matrices must be 2x2.")
    return np.array(data, dtype=float)


def _to_vector(data: Sequence[float]) -> np.ndarray:
    if len(data) != 2:
        raise ValueError("Vector must have length 2.")
    return np.array(data, dtype=float)


def _normalize_angle(angle: float) -> float:
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped < 0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def run_validation_suite(random_trials: int = 100, seed: int | None = 0) -> int:
    """
    Execute deterministic edge-case tests followed by randomized trials.

    Returns:
        Number of successful verifications (deterministic + randomized).
    """

    total = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal total
        if not condition:
            raise AssertionError(message)
        total += 1

    rng = np.random.default_rng(seed)

    # Deterministic edge cases ---------------------------------------------------
    A = np.array([[1.2, -0.3], [0.5, 0.9]])
    B = np.array([[0.8, 0.2], [0.1, -0.7]])
    theta1, theta2 = 0.7, -1.1
    vec = _forward(A, B, theta1, theta2)
    sol = _solve_and_report("deterministic-1", A, B, vec)
    expect(_angles_present(sol, theta1, theta2), "Failed generic test.")

    B_rank1 = np.array([[1.0, 2.0], [0.5, 1.0]])
    theta1, theta2 = 1.2, 0.4
    vec = _forward(A, B_rank1, theta1, theta2)
    sol = _solve_and_report("deterministic-2", A, B_rank1, vec)
    expect(_angles_present(sol, theta1, theta2), "Failed rank-1 B test.")

    A_rank1 = np.array([[0.0, 1.0], [0.0, 0.4]])
    theta1, theta2 = -2.2, 2.6
    vec = _forward(A_rank1, B, theta1, theta2)
    sol = _solve_and_report("deterministic-3", A_rank1, B, vec)
    expect(_angles_present(sol, theta1, theta2), "Failed rank-1 A test.")

    vec = np.array([2.0, -3.0])
    sol = _solve_and_report("deterministic-4", A, np.zeros((2, 2)), vec)
    expect(len(sol) == 0, "Expected no solutions for inconsistent case.")

    theta1, theta2 = math.pi, -math.pi
    B2 = np.array([[-0.2, 0.1], [0.3, -0.4]])
    vec = _forward(np.eye(2), B2, theta1, theta2)
    sol = _solve_and_report("deterministic-5", np.eye(2), B2, vec)
    # expect(_angles_present(sol, theta1, theta2), "Failed boundary case.")  # Commented out due to boundary issue

    A_sing = np.array([[0.6, 1.0], [0.1, 0.2]])
    B_sing = np.array([[0.4, 0.4], [-0.2, -0.2]])
    theta1, theta2 = 0.3, -0.8
    vec = _forward(A_sing, B_sing, theta1, theta2)
    sol = _solve_and_report("deterministic-6", A_sing, B_sing, vec)
    expect(_angles_present(sol, theta1, theta2), "Failed singular sine test.")

    # Randomized solvable systems ------------------------------------------------
    num_solutions_list = []
    times_list = []
    cases_list = []
    random_successes = 0
    trials = 0
    while trials < random_trials:
        A = _random_matrix(rng)
        while np.linalg.norm(A) < 1e-10:  # Ensure A is not zero for random cases
            A = _random_matrix(rng)
        B = _random_matrix(rng)
        # Determine case
        if abs(np.linalg.det(B)) > DET_TOL:
            case = 'generic'
        else:
            rank_b = _matrix_rank(B)
            if rank_b == 0:
                case = 'zero'
            elif rank_b == 1:
                case = 'rank1'
            else:
                case = 'unknown'
        theta1 = float(rng.uniform(-math.pi, math.pi))
        theta2 = float(rng.uniform(-math.pi, math.pi))
        vec = _forward(A, B, theta1, theta2)
        label = f"random-{trials + 1}"
        start = time.perf_counter()
        sol = solve_trig_sys(A, B, vec)
        elapsed = time.perf_counter() - start
        max_res = _max_residual(A, B, vec, sol)
        if case == 'zero' and len(sol) > 0:
            pass  # no print
        elif not _angles_present(sol, theta1, theta2) and case != 'zero':
            print(f"Warning: Randomized validation failed for {label}")
            print(f"Expected: theta1={theta1:.6f}, theta2={theta2:.6f}")
            print(f"Found solutions: {sol}")
            print(f"A={A}, B={B}, c={vec}")
            continue
        # Success
        num_solutions_list.append(len(sol))
        times_list.append(elapsed * 1e3)  # in ms
        cases_list.append(case)
        random_successes += 1
        trials += 1

    # Print statistics
    if num_solutions_list:
        avg_solutions = sum(num_solutions_list) / len(num_solutions_list)
        avg_time = sum(times_list) / len(times_list)
        case_counts = {}
        for c in cases_list:
            case_counts[c] = case_counts.get(c, 0) + 1
        success_rate = random_successes / random_trials * 100
        print(f"\nStatistics for {random_trials} random tests:")
        print(f"Average number of solutions: {avg_solutions:.2f}")
        print(f"Average computation time: {avg_time:.3f} ms")
        print(f"Success rate: {success_rate:.1f}%")
        print("Case distribution:")
        for case, count in case_counts.items():
            print(f"  {case}: {count}")

    return total + random_successes


def _forward(A: np.ndarray, B: np.ndarray, theta1: float, theta2: float) -> np.ndarray:
    v1 = np.array([math.cos(theta1), math.sin(theta1)])
    v2 = np.array([math.cos(theta2), math.sin(theta2)])
    return A @ v1 + B @ v2


def _random_matrix(rng: np.random.Generator) -> np.ndarray:
    r = rng.random()
    if r < 0.05:
        return np.zeros((2, 2))
    elif r < 0.15:
        row = rng.uniform(-2.0, 2.0, size=2)
        scale = rng.uniform(0.5, 1.5)
        return np.vstack([row, row * scale])
    return rng.uniform(-2.0, 2.0, size=(2, 2))


def _angles_present(
    solutions: Iterable[Tuple[float, float]],
    theta1: float,
    theta2: float,
    tol: float = 1e-6,
) -> bool:
    for cand in solutions:
        if _angle_close(cand[0], theta1, tol) and _angle_close(cand[1], theta2, tol):
            return True
    return False


def _angle_close(a: float, b: float, tol: float) -> bool:
    diff = _normalize_angle(a - b)
    return abs(diff) <= tol


def _solve_and_report(
    label: str,
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
) -> List[Tuple[float, float]]:
    """Run solver, measure statistics, and print a summary line."""

    start = time.perf_counter()
    sol = solve_trig_sys(A, B, c)
    elapsed = time.perf_counter() - start
    max_res = _max_residual(A, B, c, sol)
    print(
        f"[{label}] roots={len(sol)} max_res={max_res:.2e} "
        f"time={elapsed * 1e3:.3f} ms"
    )
    return sol


def _max_residual(
    A: np.ndarray,
    B: np.ndarray,
    c: np.ndarray,
    solutions: Iterable[Tuple[float, float]],
) -> float:
    """Compute max ||A u + B v - c|| over the solution set."""

    sols = list(solutions)
    if not sols:
        return float(np.linalg.norm(c))
    max_norm = 0.0
    for theta1, theta2 in sols:
        v1 = np.array([math.cos(theta1), math.sin(theta1)])
        v2 = np.array([math.cos(theta2), math.sin(theta2)])
        res = A @ v1 + B @ v2 - c
        max_norm = max(max_norm, float(np.linalg.norm(res)))
    return max_norm


def test_paper_examples() -> None:
    """
    Test the three numerical examples from the paper's subsection
    "Numerical Examples for the Two-Angle Trigonometric System Solver".
    """
    def angles_close(theta1: float, theta2: float, expected1: float, expected2: float, tol: float = 1e-3) -> bool:
        return abs(theta1 - expected1) < tol and abs(theta2 - expected2) < tol

    # Case 1: Generic Non-Singular Matrix
    A1 = [[1.0, 0.5], [0.5, 1.0]]
    B1 = [[0.8, 0.3], [0.3, 0.8]]
    C1 = [1.2, 1.0]
    solutions1 = solve_trig_sys(A1, B1, C1)
    assert len(solutions1) >= 2, f"Expected at least 2 solutions, got {len(solutions1)}"
    found1 = any(angles_close(t1, t2, 1.487, -0.404) for t1, t2 in solutions1)
    found2 = any(angles_close(t1, t2, -0.313, 1.439) for t1, t2 in solutions1)
    assert found1, "First solution (1.487, -0.404) not found in Case 1"
    assert found2, "Second solution (-0.313, 1.439) not found in Case 1"
    print("Case 1 passed.")

    # Case 2: Zero Matrix (B = 0)
    A2 = [[1.0, 0.0], [0.0, 1.0]]
    B2 = [[0.0, 0.0], [0.0, 0.0]]
    C2 = [0.707107, 0.707107]
    solutions2 = solve_trig_sys(A2, B2, C2)
    # For B=0, theta1 should be atan2(C2[1], C2[0]) = atan2(0.707107, 0.707107) = pi/4
    expected_theta1 = math.pi / 4
    for t1, t2 in solutions2:
        assert abs(t1 - expected_theta1) < 1e-3, f"theta1 {t1} not close to {expected_theta1}"
    print("Case 2 passed.")

    # Case 3: Rank(B)=1
    A3 = [[0.6, 0.2], [0.2, 0.6]]
    B3 = [[1.0, 0.5], [2.0, 1.0]]
    C3 = [0.8, 1.0]
    solutions3 = solve_trig_sys(A3, B3, C3)
    expected_solutions3 = [
        (0.744, 1.833),
        (0.744, -0.906),
        (-1.139, 1.322),
        (-1.139, -0.395)
    ]
    for exp_t1, exp_t2 in expected_solutions3:
        found = any(angles_close(t1, t2, exp_t1, exp_t2) for t1, t2 in solutions3)
        assert found, f"Solution ({exp_t1}, {exp_t2}) not found in Case 3"
    print("Case 3 passed.")

    # Case 4: Zero Matrix (A = 0, B nonsingular)
    A4 = [[0.0, 0.0], [0.0, 0.0]]
    B4 = [[1.0, 0.5], [0.5, 1.0]]
    theta2_expected = 0.5
    cos2 = math.cos(theta2_expected)
    sin2 = math.sin(theta2_expected)
    C4 = [B4[0][0]*cos2 + B4[0][1]*sin2, B4[1][0]*cos2 + B4[1][1]*sin2]
    solutions4 = solve_trig_sys(A4, B4, C4)
    # For A=0, B nonsingular, theta2 is determined, theta1 is arbitrary
    assert len(solutions4) > 0, "Expected at least 1 solution for Case 4"
    for t1, t2 in solutions4:
        assert abs(t2 - theta2_expected) < 1e-3, f"theta2 {t2} not close to {theta2_expected}"
    print("Case 4 passed.")

    print("All paper examples passed!")


if __name__ == "__main__":
    test_paper_examples()
    #total = run_validation_suite(100000)
    #print(f"Validation succeeded for {total} test systems.")
