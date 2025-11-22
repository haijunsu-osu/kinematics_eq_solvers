"""
Single-angle trigonometric solver derived from the linear-unit-circle method.

Given a 2×2 matrix ``A`` and vector ``c`` the system

    A @ [cos(theta), sin(theta)]^T = c

either has no solution, a unique solution, or—when ``A`` is rank deficient—an
infinite family. The solver proceeds as follows:

* If ``A`` is nonsingular we solve for ``v = A^{-1} c`` using ``np.linalg.solve``,
  enforce the unit-circle constraint (``||v|| = 1``), and convert ``v`` to the
  angle via ``atan2``. Residuals are checked against the original equations.
* If ``A`` has rank one we select the row with the largest coefficient norm and
  solve the single trigonometric equation ``a cos(theta) + b sin(theta) = d``.
  Up to two candidate angles arise from the phase-shift identity. Each is
  validated with the full linear system, so only compatible solutions remain.
* If ``A`` has rank zero we detect whether the system is inconsistent or admits
  arbitrary solutions (``c = 0``). In the arbitrary case the solver returns a
  zero angle with a warning flag.

The public API returns ``{'solutions': [...], 'warning': ''}`` (or
``'arbitrary solution'`` when the system is underdetermined). All reported
angles are normalized to ``[-pi, pi]`` and verified within strict tolerances.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple, Union

import numpy as np


NORM_TOL = 1e-8
RES_TOL = 1e-10
NUM_TOL = 1e-10
DET_TOL = 1e-10


ResultDict = Dict[str, Union[List[float], str]]


def solve_trig_sys_single(A: np.ndarray, c: np.ndarray) -> ResultDict:
    """
    Solve ``A @ [cos(theta), sin(theta)] = c`` for theta.

    Args:
        A: ``2x2`` real matrix.
        c: length-2 vector.

    Returns:
        Dictionary with keys:
            ``solutions``: list of normalized angles (possibly empty).
            ``warning``: empty string when unique, ``'arbitrary solution'`` when
                the system has infinitely many solutions.

    Example:
        >>> import math, numpy as np
        >>> theta = 0.8
        >>> A = np.array([[2.0, -0.6], [0.5, 1.3]])
        >>> c = A @ np.array([math.cos(theta), math.sin(theta)])
        >>> solve_trig_sys_single(A, c)
        {'solutions': [0.8], 'warning': ''}
    """

    mat = _to_matrix(A)
    vec = _to_vector(c)
    rank = _matrix_rank(mat)

    if rank == 2:
        return _solve_full_rank(mat, vec)
    if rank == 1:
        return _solve_rank1(mat, vec)
    return _solve_rank0(mat, vec)


def _solve_full_rank(A: np.ndarray, c: np.ndarray) -> ResultDict:
    try:
        sol = np.linalg.solve(A, c)
    except np.linalg.LinAlgError:
        return {"solutions": [], "warning": ""}

    norm = np.linalg.norm(sol)
    if abs(norm - 1.0) > NORM_TOL or norm < NUM_TOL:
        return {"solutions": [], "warning": ""}
    sol /= norm
    theta = _normalize_angle(math.atan2(sol[1], sol[0]))
    if _verify(A, c, theta):
        return {"solutions": [theta], "warning": ""}
    return {"solutions": [], "warning": ""}


def _solve_rank1(A: np.ndarray, c: np.ndarray) -> ResultDict:
    row, target = _dominant_row_and_target(A, c)
    if row is None:
        return {"solutions": [], "warning": ""}
    candidates = _solve_trig_eq(row[0], row[1], target)
    solutions: List[float] = []
    for theta in candidates:
        theta = _normalize_angle(theta)
        if _verify(A, c, theta) and not _angle_in_list(theta, solutions):
            solutions.append(theta)
    solutions.sort()
    return {"solutions": solutions, "warning": ""}


def _solve_rank0(A: np.ndarray, c: np.ndarray) -> ResultDict:
    if np.linalg.norm(c) <= NUM_TOL:
        return {"solutions": [0.0], "warning": "arbitrary solution"}
    return {"solutions": [], "warning": ""}


def _solve_trig_eq(a: float, b: float, d: float) -> List[float]:
    """Solve a cos(theta) + b sin(theta) = d for theta."""

    radius = math.hypot(a, b)
    if radius < NUM_TOL:
        return []
    ratio = d / radius
    if abs(ratio) > 1.0 + 1e-9:
        return []
    ratio = max(min(ratio, 1.0), -1.0)
    delta = math.acos(ratio)
    phase = math.atan2(b, a)
    return [phase + delta, phase - delta]


def _verify(A: np.ndarray, c: np.ndarray, theta: float) -> bool:
    vec = np.array([math.cos(theta), math.sin(theta)])
    residual = A @ vec - c
    cond = np.linalg.cond(A)
    if not np.isfinite(cond):
        cond = 1.0 / max(1e-12, np.linalg.norm(A))
    tol = RES_TOL * (1.0 + np.linalg.norm(A)) * max(1.0, cond)
    return float(np.linalg.norm(residual)) <= tol


def _matrix_rank(A: np.ndarray) -> int:
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > DET_TOL))


def _dominant_row_and_target(
    A: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray | None, float]:
    norms = [np.linalg.norm(A[0]), np.linalg.norm(A[1])]
    idx = int(np.argmax(norms))
    if norms[idx] < NUM_TOL:
        return None, 0.0
    return A[idx], float(c[idx])


def _angle_in_list(theta: float, collection: List[float], tol: float = 1e-10) -> bool:
    return any(abs(_normalize_angle(theta - existing)) <= tol for existing in collection)


def _normalize_angle(angle: float) -> float:
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped < 0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def _to_matrix(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError("A must be 2x2.")
    return arr


def _to_vector(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float).reshape(-1)
    if arr.shape != (2,):
        raise ValueError("c must be length 2.")
    return arr


def run_validation_suite(random_trials: int = 1000, seed: int | None = 0) -> int:
    """
    Execute deterministic edge tests followed by randomized solvable systems.

    Returns:
        Number of successful verifications. Raises AssertionError on failure.
    """

    total = 0

    def expect(condition: bool, message: str) -> None:
        nonlocal total
        if not condition:
            raise AssertionError(message)
        print(f"ASSERTION: {message}")
        total += 1

    rng = np.random.default_rng(seed)

    # Invertible system at a general angle.
    theta = 0.73
    A = np.array([[1.4, -0.2], [0.5, 1.1]])
    c = A @ np.array([math.cos(theta), math.sin(theta)])
    res = _solve_and_report("deterministic-1 (full rank)", A, c)
    expect(_angles_present(res["solutions"], theta), "Recovered angle for invertible case.")

    # Norm mismatch (inconsistent).
    res = _solve_and_report("deterministic-2 (norm mismatch)", np.eye(2), np.array([2.0, 0.0]))
    expect(res["solutions"] == [], "Detected inconsistent norm mismatch.")

    # Boundary theta = 0.
    res = _solve_and_report("deterministic-3 (theta=0)", np.eye(2), np.array([1.0, 0.0]))
    expect(_angles_present(res["solutions"], 0.0), "Solved theta=0 boundary.")

    # Boundary theta = pi.
    res = _solve_and_report("deterministic-4 (theta=pi)", np.eye(2), np.array([-1.0, 0.0]))
    expect(_angles_present(res["solutions"], math.pi), "Solved theta=pi boundary.")

    # Rank-1 consistent system (two dependent rows).
    rank1 = np.array([[1.0, 2.0], [2.0, 4.0]])
    theta = -0.9
    c = rank1 @ np.array([math.cos(theta), math.sin(theta)])
    res = _solve_and_report("deterministic-5 (rank-1 consistent)", rank1, c)
    expect(_angles_present(res["solutions"], theta), "Solved rank-1 consistent case.")

    # Rank-1 inconsistent system.
    res = _solve_and_report("deterministic-6 (rank-1 inconsistent)", rank1, np.array([5.0, 10.0]))
    expect(res["solutions"] == [], "Detected rank-1 inconsistency.")

    # Rank-0 arbitrary solution.
    res = _solve_and_report("deterministic-7 (rank-0 arbitrary)", np.zeros((2, 2)), np.zeros(2))
    expect(res["warning"] == "arbitrary solution", "Flagged arbitrary solution.")

    # Rank-0 inconsistent.
    res = _solve_and_report("deterministic-8 (rank-0 inconsistent)", np.zeros((2, 2)), np.array([0.1, 0.0]))
    expect(res["solutions"] == [], "Detected rank-0 inconsistency.")

    # Nearly singular but solvable matrix.
    eps = 1e-8
    near = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
    theta = 1.2
    c = near @ np.array([math.cos(theta), math.sin(theta)])
    res = _solve_and_report("deterministic-9 (near singular)", near, c)
    expect(_angles_present(res["solutions"], theta, tol=5e-8), "Solved nearly singular case.")

    # Randomized solvable systems.
    trials = 0
    while trials < random_trials:
        A = _random_invertible_matrix(rng)
        theta = float(rng.uniform(-math.pi, math.pi))
        c = A @ np.array([math.cos(theta), math.sin(theta)])
        label = f"random-{trials + 1}"
        res = _solve_and_report(label, A, c)
        if not _angles_present(res["solutions"], theta, tol=1e-9):
            raise AssertionError("Random validation failed.")
        total += 1
        trials += 1

    return total


def _solve_and_report(label: str, A: np.ndarray, c: np.ndarray) -> ResultDict:
    """Run the solver, time it, and report root statistics."""

    start = time.perf_counter()
    result = solve_trig_sys_single(A, c)
    elapsed = time.perf_counter() - start
    max_res = _max_residual(A, c, result["solutions"])
    warning = result["warning"]
    print(
        f"[{label}] roots={len(result['solutions'])} "
        f"max_res={max_res:.2e} time={elapsed * 1e3:.3f} ms "
        f"warning={'none' if not warning else warning}"
    )
    return result


def _max_residual(A: np.ndarray, c: np.ndarray, solutions: List[float]) -> float:
    """Return the maximum residual norm over provided solutions."""

    if not solutions:
        return float(np.linalg.norm(c))
    max_norm = 0.0
    for theta in solutions:
        vec = np.array([math.cos(theta), math.sin(theta)])
        res = A @ vec - c
        max_norm = max(max_norm, float(np.linalg.norm(res)))
    return max_norm


def _angles_present(solutions: List[float], theta: float, tol: float = 1e-10) -> bool:
    return any(abs(_normalize_angle(sol - theta)) <= tol for sol in solutions)


def _random_invertible_matrix(rng: np.random.Generator) -> np.ndarray:
    while True:
        mat = rng.uniform(-2.0, 2.0, size=(2, 2))
        if abs(np.linalg.det(mat)) > DET_TOL:
            return mat


if __name__ == "__main__":
    total = run_validation_suite()
    print(f"Validation succeeded for {total} test systems.")
