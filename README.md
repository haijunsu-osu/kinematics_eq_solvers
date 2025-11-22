# Kinematics Equation Solvers

This repository provides analytical solvers for common algebraic equations that arise in robot kinematics problems. These solvers are designed to handle trigonometric and bilinear systems efficiently, enumerating all real solutions while managing singularities and degeneracies.

## Overview

The solvers address four primary types of equations encountered in inverse kinematics:

1. **Single Trigonometric Equation**: $a \cos \theta + b \sin \theta + c = 0$
2. **Single-Angle Trigonometric System**: $A [\cos \theta, \sin \theta]^T = \mathbf{c}$
3. **Two-Angle Trigonometric System**: $A [\cos \theta_1, \sin \theta_1]^T + B [\cos \theta_2, \sin \theta_2]^T = \mathbf{c}$
4. **Bilinear Two-Angle System**: $K \mathbf{m} = \mathbf{0}$, where $\mathbf{m} = (1, c_1, s_1, c_2, s_2, c_1 c_2, c_1 s_2, s_1 c_2, s_1 s_2)^T$

Each solver uses analytical methods (e.g., Weierstrass substitution, resultant elimination) to avoid numerical instabilities and ensure completeness of solutions.

## Features

- **Robust Degeneracy Handling**: Manages singular matrices, free parameters, and inconsistent systems.
- **Complete Solution Enumeration**: Finds all real solutions up to the maximum possible (e.g., 8 for bilinear systems).
- **High Accuracy**: Solutions validated to machine precision.
- **Python Implementation**: Self-contained functions using only standard libraries.
- **LLM-Generated Code**: All solvers were generated using OpenAI's Codex 5.1 via carefully crafted prompts.

## Repository Structure

- `src/`: Python source code for the solvers.
  - `solve_trig_eq_solver.py`: Single trigonometric equation solver.
  - `solve_trig_sys_single_solver.py`: Single-angle trigonometric system solver.
  - `solve_trig_sys_solver.py`: Two-angle trigonometric system solver.
  - `solve_bilinear_sys_solver.py`: Bilinear two-angle system solver.
- `prompts/`: Text files containing the prompts used to generate the code via Codex 5.1.
  - `prompt_solve_trig_eq.txt`
  - `prompt_solve_trig_sys_single.txt`
  - `prompt_solve_trig_sys.txt`
- `paper/eq_solvers.pdf`: Compiled PDF of the paper.
- Other LaTeX auxiliary files.

## Usage

Each solver is a standalone Python function. Import and use as follows:

```python
from src.solve_trig_eq_solver import solve_trig_eq

# Example: Solve a*cos(theta) + b*sin(theta) + c = 0
solutions = solve_trig_eq(a=1.0, b=2.0, c=0.5)
print(solutions)  # List of angles in [-pi, pi]
```

Refer to the paper for detailed mathematical derivations, examples, and benchmarks. The derivation of the solution process is given in the PDF file in the `paper/` folder.

## Generating Code via LLM

The code in `src/` was generated using OpenAI's Codex 5.1 coding agent. The prompts in `prompts/` were crafted to produce robust, production-ready implementations. Each prompt includes:

- Context and task description.
- Requirements for handling degeneracies, validation, and documentation.
- Deliverables specification.

To reproduce or modify, submit the prompt text to a compatible LLM (e.g., Codex 5.1) and refine as needed.

## Benchmarks

The paper includes extensive benchmarks:
- 100,000 random tests for trigonometric systems (100% success rate, ~0.6 ms average).
- Validation on degenerate cases and up to 8 solutions for bilinear systems.

## Citation

If you use this work, please cite the paper:

```
@article{su2025analytical,
  title={Analytical Solvers for Common Algebraic Equations Arising in Kinematics Problems},
  author={Su, Hai-Jun},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This repository is provided for educational and research purposes. See the paper for any specific licensing details.

## Contact

Hai-Jun Su (su.298@osu.edu)

Department of Mechanical and Aerospace Engineering, The Ohio State University