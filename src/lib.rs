/*!
A library for solving linear programming problems.

[Linear programming](https://en.wikipedia.org/wiki/Linear_programming) is technique for
finding the minimum (or maximum) of a linear function of a set of continuous variables
subject to linear equality and inequality constraints.

# Features

* Pure Rust implementation.
* Able to solve problems with hundreds of thousands of variables and constraints.
* Incremental: add constraints to an existing solution without solving it from scratch.
* Problems can be defined via an API or parsed from an
  [MPS](https://en.wikipedia.org/wiki/MPS_(format)) file.

# Entry points

Begin by creating a [`Problem`](struct.Problem.html) instance, declaring variables and adding
constraints. Solving it will produce [`Solution`](struct.Solution.html) that can be used to
get the optimal objective value, corresponding variable values and also to add more
constraints.

Alternatively, use the [`mps`](io/index.html) parser to read the problem from an MPS file.

# Example

```
use minilp::{Problem, OptimizationDirection, ComparisonOp};

// Maximize an objective function x + 2 * y of two variables x >= 0 and 0 <= y <= 3
let mut problem = Problem::new(OptimizationDirection::Maximize);
let x = problem.add_var((0.0, f64::INFINITY), 1.0);
let y = problem.add_var((0.0, 3.0), 2.0);

// subject to constraints: x + y <= 4 and 2 * x + y >= 2.0.
problem.add_constraint(&[(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
problem.add_constraint(&[(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

// Optimal value is 7, achieved at x = 1 and y = 3.
let solution = problem.solve().unwrap();
assert_eq!(solution.objective(), 7.0);
assert_eq!(solution[x], 1.0);
assert_eq!(solution[y], 3.0);
```
*/

#[macro_use]
extern crate log;

mod helpers;
pub mod io;
mod lu;
mod ordering;
mod solver;
mod sparse;

use solver::Solver;

#[derive(Clone, Copy, Debug)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(pub(crate) usize);

impl Variable {
    pub fn idx(&self) -> usize {
        self.0
    }
}

#[derive(Clone)]
pub struct LinearExpr {
    vars: Vec<usize>,
    coeffs: Vec<f64>,
}

impl LinearExpr {
    pub fn empty() -> Self {
        Self {
            vars: vec![],
            coeffs: vec![],
        }
    }

    pub fn add(&mut self, var: Variable, coeff: f64) {
        self.vars.push(var.0);
        self.coeffs.push(coeff);
    }
}

pub struct LinearTerm(Variable, f64);

impl From<(Variable, f64)> for LinearTerm {
    fn from(term: (Variable, f64)) -> Self {
        LinearTerm(term.0, term.1)
    }
}

impl<'a> From<&'a (Variable, f64)> for LinearTerm {
    fn from(term: &'a (Variable, f64)) -> Self {
        LinearTerm(term.0, term.1)
    }
}

impl<I: IntoIterator<Item = impl Into<LinearTerm>>> From<I> for LinearExpr {
    fn from(iter: I) -> Self {
        let mut expr = LinearExpr::empty();
        for term in iter {
            let LinearTerm(var, coeff) = term.into();
            expr.add(var, coeff);
        }
        expr
    }
}

impl std::iter::FromIterator<(Variable, f64)> for LinearExpr {
    fn from_iter<I: IntoIterator<Item = (Variable, f64)>>(iter: I) -> Self {
        let mut expr = LinearExpr::empty();
        for term in iter {
            expr.add(term.0, term.1)
        }
        expr
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ComparisonOp {
    /// The == operator (equal to)
    Eq,
    /// The <= operator (less than or equal to)
    Le,
    /// The >= operator (greater than or equal to)
    Ge,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    Infeasible,
    Unbounded,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self {
            Error::Infeasible => "problem is infeasible",
            Error::Unbounded => "problem is unbounded",
        };
        msg.fmt(f)
    }
}

impl std::error::Error for Error {}

#[derive(Clone)]
pub struct Problem {
    direction: OptimizationDirection,
    obj_coeffs: Vec<f64>,
    var_mins: Vec<f64>,
    var_maxs: Vec<f64>,
    constraints: Vec<(CsVec, ComparisonOp, f64)>,
}

type CsVec = sprs::CsVecI<f64, usize>;

impl Problem {
    pub fn new(direction: OptimizationDirection) -> Self {
        Problem {
            direction,
            obj_coeffs: vec![],
            var_mins: vec![],
            var_maxs: vec![],
            constraints: vec![],
        }
    }

    pub fn add_var(&mut self, (min, max): (f64, f64), obj_coeff: f64) -> Variable {
        let var = Variable(self.obj_coeffs.len());
        let obj_coeff = match self.direction {
            OptimizationDirection::Minimize => obj_coeff,
            OptimizationDirection::Maximize => -obj_coeff,
        };
        self.obj_coeffs.push(obj_coeff);
        self.var_mins.push(min);
        self.var_maxs.push(max);
        var
    }

    pub fn add_constraint(&mut self, expr: impl Into<LinearExpr>, cmp_op: ComparisonOp, rhs: f64) {
        let expr = expr.into();
        self.constraints.push((
            CsVec::new(self.obj_coeffs.len(), expr.vars, expr.coeffs),
            cmp_op,
            rhs,
        ));
    }

    pub fn solve(&self) -> Result<Solution, Error> {
        let mut solver = Solver::try_new(
            &self.obj_coeffs,
            &self.var_mins,
            &self.var_maxs,
            &self.constraints,
        )?;
        solver.optimize()?;
        Ok(Solution {
            num_vars: self.obj_coeffs.len(),
            direction: self.direction,
            solver,
        })
    }
}

#[derive(Clone)]
pub struct Solution {
    num_vars: usize,
    direction: OptimizationDirection,
    solver: solver::Solver,
}

impl Solution {
    pub fn objective(&self) -> f64 {
        match self.direction {
            OptimizationDirection::Minimize => self.solver.cur_obj_val,
            OptimizationDirection::Maximize => -self.solver.cur_obj_val,
        }
    }

    pub fn get(&self, var: Variable) -> &f64 {
        assert!(var.idx() < self.num_vars);
        self.solver.get_value(var.idx())
    }

    pub fn iter(&self) -> SolutionIterator {
        SolutionIterator {
            solution: self,
            var_idx: 0,
        }
    }

    pub fn set_var(mut self, var: Variable, val: f64) -> Result<Self, Error> {
        assert!(var.idx() < self.num_vars);
        self.solver.set_var(var.idx(), val)?;
        Ok(self)
    }

    /// Return true if the var was really unset.
    pub fn unset_var(mut self, var: Variable) -> Result<(Self, bool), Error> {
        assert!(var.idx() < self.num_vars);
        let res = self.solver.unset_var(var.idx())?;
        Ok((self, res))
    }

    pub fn add_constraint(
        mut self,
        expr: impl Into<LinearExpr>,
        cmp_op: ComparisonOp,
        rhs: f64,
    ) -> Result<Self, Error> {
        let expr = expr.into();
        self.solver.add_constraint(
            CsVec::new(self.num_vars, expr.vars, expr.coeffs),
            cmp_op,
            rhs,
        )?;
        Ok(self)
    }

    // TODO: remove_constraint

    pub fn add_gomory_cut(mut self, var: Variable) -> Result<Self, Error> {
        assert!(var.idx() < self.num_vars);
        self.solver.add_gomory_cut(var.idx())?;
        Ok(self)
    }
}

impl std::ops::Index<Variable> for Solution {
    type Output = f64;

    fn index(&self, var: Variable) -> &Self::Output {
        self.get(var)
    }
}

pub struct SolutionIterator<'a> {
    solution: &'a Solution,
    var_idx: usize,
}

impl<'a> Iterator for SolutionIterator<'a> {
    type Item = (Variable, &'a f64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.var_idx < self.solution.num_vars {
            let var_idx = self.var_idx;
            self.var_idx += 1;
            Some((Variable(var_idx), self.solution.solver.get_value(var_idx)))
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Solution {
    type Item = (Variable, &'a f64);
    type IntoIter = SolutionIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize() {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var((12.0, f64::INFINITY), 3.0);
        let v2 = problem.add_var((5.0, f64::INFINITY), 4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint(&[(v2, -4.0), (v1, 1.0)], ComparisonOp::Ge, -20.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 12.0);
        assert_eq!(sol[v2], 8.0);
        assert_eq!(sol.objective(), 68.0);
    }

    #[test]
    fn empty_expr_constraints() {
        let trivial = [
            (LinearExpr::empty(), ComparisonOp::Eq, 0.0),
            (LinearExpr::empty(), ComparisonOp::Ge, -1.0),
            (LinearExpr::empty(), ComparisonOp::Le, 1.0),
        ];

        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let _ = problem.add_var((0.0, f64::INFINITY), 1.0);
        for (expr, op, b) in trivial.iter().cloned() {
            problem.add_constraint(expr, op, b);
        }
        assert_eq!(problem.solve().map(|s| s.objective()), Ok(0.0));

        {
            let mut sol = problem.solve().unwrap();
            for (expr, op, b) in trivial.iter().cloned() {
                sol = sol.add_constraint(expr, op, b).unwrap();
            }
            assert_eq!(sol.objective(), 0.0);
        }

        let infeasible = [
            (LinearExpr::empty(), ComparisonOp::Eq, 12.0),
            (LinearExpr::empty(), ComparisonOp::Ge, 34.0),
            (LinearExpr::empty(), ComparisonOp::Le, -56.0),
        ];

        for (expr, op, b) in infeasible.iter().cloned() {
            let mut cloned = problem.clone();
            cloned.add_constraint(expr, op, b);
            assert_eq!(cloned.solve().map(|_| "solved"), Err(Error::Infeasible));
        }

        for (expr, op, b) in infeasible.iter().cloned() {
            let sol = problem.solve().unwrap().add_constraint(expr, op, b);
            assert_eq!(sol.map(|_| "solved"), Err(Error::Infeasible));
        }

        let _ = problem.add_var((0.0, f64::INFINITY), -1.0);
        assert_eq!(problem.solve().map(|_| "solved"), Err(Error::Unbounded));
    }

    #[test]
    fn set_unset_var() {
        let mut problem = Problem::new(OptimizationDirection::Maximize);
        let v1 = problem.add_var((0.0, 3.0), 1.0);
        let v2 = problem.add_var((0.0, 3.0), 2.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 1.0);

        let orig_sol = problem.solve().unwrap();

        {
            let mut sol = orig_sol.clone().set_var(v1, 0.5).unwrap();
            assert_eq!(sol[v1], 0.5);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unset_var(v1).unwrap().0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }

        {
            let mut sol = orig_sol.clone().set_var(v2, 2.5).unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 2.5);
            assert_eq!(sol.objective(), 6.5);

            sol = sol.unset_var(v2).unwrap().0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 7.0);
        }
    }

    #[test]
    fn add_constraint() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var((0.0, f64::INFINITY), 2.0);
        let v2 = problem.add_var((0.0, f64::INFINITY), 1.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], ComparisonOp::Ge, 2.0);

        let orig_sol = problem.solve().unwrap();

        {
            let sol = orig_sol
                .clone()
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();

            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 1.0);
            assert_eq!(sol.objective(), 3.0);
        }

        {
            let sol = orig_sol
                .clone()
                .set_var(v2, 1.5)
                .unwrap()
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], ComparisonOp::Le, 0.0)
                .unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 1.5);
            assert_eq!(sol.objective(), 4.5);
        }

        {
            let sol = orig_sol
                .clone()
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], ComparisonOp::Ge, 3.0)
                .unwrap();

            assert_eq!(sol[v1], 0.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 3.0);
        }
    }

    #[test]
    fn gomory_cut() {
        let mut problem = Problem::new(OptimizationDirection::Minimize);
        let v1 = problem.add_var((0.0, f64::INFINITY), 0.0);
        let v2 = problem.add_var((0.0, f64::INFINITY), -1.0);
        problem.add_constraint(&[(v1, 3.0), (v2, 2.0)], ComparisonOp::Le, 6.0);
        problem.add_constraint(&[(v1, -3.0), (v2, 2.0)], ComparisonOp::Le, 0.0);

        let mut sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 1.0);
        assert_eq!(sol[v2], 1.5);
        assert_eq!(sol.objective(), -1.5);

        sol = sol.add_gomory_cut(v2).unwrap();
        assert!(f64::abs(sol[v1] - 2.0 / 3.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);

        sol = sol.add_gomory_cut(v1).unwrap();
        assert!(f64::abs(sol[v1] - 1.0) < 1e-8);
        assert_eq!(sol[v2], 1.0);
        assert_eq!(sol.objective(), -1.0);
    }
}
