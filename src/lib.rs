#[macro_use]
extern crate log;

mod helpers;
mod lu;
mod ordering;
mod solver;
mod sparse;

use solver::Solver;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(usize);

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
pub enum RelOp {
    Eq,
    Le,
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
    obj: Vec<f64>,
    var_mins: Vec<f64>,
    var_maxs: Vec<f64>,
    constraints: Vec<(CsVec, RelOp, f64)>,
}

type CsVec = sprs::CsVecI<f64, usize>;

impl Problem {
    pub fn new() -> Self {
        Problem {
            obj: vec![],
            var_mins: vec![],
            var_maxs: vec![],
            constraints: vec![],
        }
    }

    pub fn add_var(&mut self, min: Option<f64>, max: Option<f64>, obj_coeff: f64) -> Variable {
        let var = Variable(self.obj.len());
        self.obj.push(obj_coeff);
        self.var_mins.push(min.unwrap_or(f64::NEG_INFINITY));
        self.var_maxs.push(max.unwrap_or(f64::INFINITY));
        var
    }

    pub fn add_constraint(&mut self, expr: impl Into<LinearExpr>, rel_op: RelOp, bound: f64) {
        let expr = expr.into();
        self.constraints.push((
            CsVec::new(self.obj.len(), expr.vars, expr.coeffs),
            rel_op,
            bound,
        ));
    }

    pub fn solve(&self) -> Result<Solution, Error> {
        let mut solver = Solver::try_new(
            &self.obj,
            &self.var_mins,
            &self.var_maxs,
            &self.constraints,
        )?;
        solver.optimize()?;
        Ok(Solution {
            num_vars: self.obj.len(),
            solver,
        })
    }
}

#[derive(Clone)]
pub struct Solution {
    num_vars: usize,
    solver: solver::Solver,
}

impl Solution {
    pub fn objective(&self) -> f64 {
        self.solver.cur_obj_val
    }

    pub fn get(&self, var: Variable) -> &f64 {
        assert!(var.idx() < self.num_vars);
        self.solver.get_value(var.idx())
    }

    pub fn iter(&self) -> IterSolution {
        IterSolution {
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
        rel_op: RelOp,
        bound: f64,
    ) -> Result<Self, Error> {
        let expr = expr.into();
        self.solver.add_constraint(
            CsVec::new(self.num_vars, expr.vars, expr.coeffs),
            rel_op,
            bound,
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

pub struct IterSolution<'a> {
    solution: &'a Solution,
    var_idx: usize,
}

impl<'a> Iterator for IterSolution<'a> {
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
    type IntoIter = IterSolution<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize() {
        let mut problem = Problem::new();
        let v1 = problem.add_var(Some(12.0), None, -3.0);
        let v2 = problem.add_var(Some(5.0), None, -4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], RelOp::Le, 20.0);
        problem.add_constraint(&[(v2, -4.0), (v1, 1.0)], RelOp::Ge, -20.0);

        let sol = problem.solve().unwrap();
        assert_eq!(sol[v1], 12.0);
        assert_eq!(sol[v2], 8.0);
        assert_eq!(sol.objective(), -68.0);
    }

    #[test]
    fn empty_expr_constraints() {
        let trivial = [
            (LinearExpr::empty(), RelOp::Eq, 0.0),
            (LinearExpr::empty(), RelOp::Ge, -1.0),
            (LinearExpr::empty(), RelOp::Le, 1.0),
        ];

        let mut problem = Problem::new();
        let _ = problem.add_var(Some(0.0), None, 1.0);
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
            (LinearExpr::empty(), RelOp::Eq, 12.0),
            (LinearExpr::empty(), RelOp::Ge, 34.0),
            (LinearExpr::empty(), RelOp::Le, -56.0),
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

        let _ = problem.add_var(Some(0.0), None, -1.0);
        assert_eq!(problem.solve().map(|_| "solved"), Err(Error::Unbounded));
    }

    #[test]
    fn set_unset_var() {
        let mut problem = Problem::new();
        let v1 = problem.add_var(Some(0.0), Some(3.0), -1.0);
        let v2 = problem.add_var(Some(0.0), Some(3.0), -2.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], RelOp::Le, 4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], RelOp::Ge, 1.0);

        let orig_sol = problem.solve().unwrap();

        {
            let mut sol = orig_sol.clone().set_var(v1, 0.5).unwrap();
            assert_eq!(sol[v1], 0.5);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), -6.5);

            sol = sol.unset_var(v1).unwrap().0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), -7.0);
        }

        {
            let mut sol = orig_sol.clone().set_var(v2, 2.5).unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 2.5);
            assert_eq!(sol.objective(), -6.5);

            sol = sol.unset_var(v2).unwrap().0;
            assert_eq!(sol[v1], 1.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), -7.0);
        }
    }

    #[test]
    fn add_constraint() {
        let mut problem = Problem::new();
        let v1 = problem.add_var(Some(0.0), None, 2.0);
        let v2 = problem.add_var(Some(0.0), None, 1.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], RelOp::Le, 4.0);
        problem.add_constraint(&[(v1, 1.0), (v2, 1.0)], RelOp::Ge, 2.0);

        let orig_sol = problem.solve().unwrap();

        {
            let sol = orig_sol
                .clone()
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], RelOp::Le, 0.0)
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
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], RelOp::Le, 0.0)
                .unwrap();
            assert_eq!(sol[v1], 1.5);
            assert_eq!(sol[v2], 1.5);
            assert_eq!(sol.objective(), 4.5);
        }

        {
            let sol = orig_sol
                .clone()
                .add_constraint(&[(v1, -1.0), (v2, 1.0)], RelOp::Ge, 3.0)
                .unwrap();

            assert_eq!(sol[v1], 0.0);
            assert_eq!(sol[v2], 3.0);
            assert_eq!(sol.objective(), 3.0);
        }
    }

    #[test]
    fn gomory_cut() {
        let mut problem = Problem::new();
        let v1 = problem.add_var(Some(0.0), None, 0.0);
        let v2 = problem.add_var(Some(0.0), None, -1.0);
        problem.add_constraint(&[(v1, 3.0), (v2, 2.0)], RelOp::Le, 6.0);
        problem.add_constraint(&[(v1, -3.0), (v2, 2.0)], RelOp::Le, 0.0);

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
