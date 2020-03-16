#[macro_use]
extern crate log;

pub mod lu;
use lu::{lu_factorize, LUFactors, ScratchSpace};

pub mod ordering;

pub type CsVec = sprs::CsVecI<f64, usize>;
type CsMat = sprs::CsMatI<f64, usize>;
type ArrayVec = ndarray::Array1<f64>;

use sprs::CompressedStorage;
use std::collections::{BTreeSet, HashMap};

mod sparse;
use sparse::{ScatteredVec, SparseMat, SparseVec};

mod helpers;
use helpers::{resized_view, to_dense};

const SENTINEL: usize = 0usize.wrapping_sub(1);

#[derive(Clone)]
pub struct Tableau {
    num_vars: usize,
    num_slack_vars: usize,
    num_artificial_vars: usize,

    orig_obj: Vec<f64>,      // with negated coeffs
    orig_constraints: CsMat, // excluding bounds
    orig_constraints_csc: CsMat,
    orig_bounds: Vec<f64>,

    set_vars: HashMap<usize, f64>,

    enable_steepest_edge: bool,

    basis_solver: BasisSolver,

    // Recomputed on each pivot
    col_coeffs: SparseVec,
    eta_matrix_coeffs: SparseVec,
    sq_norms_update_helper: ScatteredVec,
    row_coeffs: ScatteredVec,

    // Updated on each pivot
    basic_vars: Vec<usize>, // for each constraint the corresponding basic var.
    basic_vars_inv: Vec<usize>, // (var -> idx if basic or sentinel) for all vars
    cur_bounds: Vec<f64>,

    non_basic_vars: Vec<usize>,     // remaining variables. (idx -> var)
    non_basic_vars_inv: Vec<usize>, // (var -> idx if non-basic or sentinel) for all vars
    cur_obj: Vec<f64>,
    non_basic_col_sq_norms: Vec<f64>,

    cur_obj_val: f64,
}

impl std::fmt::Debug for Tableau {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tableau({}, {}, {})\norig_obj:\n{:?}\n",
            self.num_vars, self.num_slack_vars, self.num_artificial_vars, self.orig_obj,
        )?;
        write!(f, "orig_constraints:\n")?;
        for row in self.orig_constraints.outer_iterator() {
            write!(f, "{:?}\n", to_dense(&row))?;
        }
        write!(f, "orig_bounds:\n{:?}\n", self.orig_bounds)?;
        write!(f, "basic_vars:\n{:?}\n", self.basic_vars)?;
        write!(f, "cur_bounds:\n{:?}\n", self.cur_bounds)?;
        write!(f, "non_basic_vars:\n{:?}\n", self.non_basic_vars)?;
        write!(f, "cur_obj:\n{:?}\n", self.cur_obj)?;
        write!(f, "cur_obj_val: {:?}\n", self.cur_obj_val)?;
        write!(f, "set_vars:\n{:?}\n", self.set_vars)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum RelOp {
    Eq,
    Le,
    Ge,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Infeasible,
    Unbounded,
}

impl Tableau {
    pub fn new(mut obj: Vec<f64>, constraints: Vec<(CsVec, RelOp, f64)>) -> Self {
        let enable_steepest_edge = true;

        let num_vars = obj.len();
        let (num_slack_vars, num_artificial_vars) = {
            let mut s = 0;
            let mut a = 0;
            for constr in &constraints {
                match constr.1 {
                    RelOp::Le => s += 1,
                    RelOp::Eq => a += 1,
                    RelOp::Ge => {
                        s += 1;
                        a += 1;
                    }
                }
            }
            (s, a)
        };
        let num_total_vars = num_vars + num_slack_vars + num_artificial_vars;
        let num_constraints = constraints.len();

        for val in &mut obj {
            *val *= -1.0;
        }
        obj.extend(std::iter::repeat(0.0).take(num_slack_vars + num_artificial_vars));

        let mut cur_slack_var = 0;
        let mut cur_artificial_var = 0;

        let mut artificial_multipliers = CsVec::empty(num_constraints);
        let mut artificial_obj_val = 0.0;

        let mut orig_constraints = CsMat::empty(CompressedStorage::CSR, num_total_vars);
        let mut orig_bounds = Vec::with_capacity(num_constraints);
        let mut basic_vars = vec![];
        let mut basic_vars_inv = vec![SENTINEL; num_total_vars];

        for (mut coeffs, rel_op, bound) in constraints.into_iter() {
            let basic_idx = match rel_op {
                RelOp::Le => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);
                    let basic_idx = cur_slack_var;
                    cur_slack_var += 1;
                    basic_idx
                }

                RelOp::Eq => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);
                    artificial_obj_val += bound;
                    artificial_multipliers.append(basic_vars.len(), 1.0);
                    let basic_idx = num_slack_vars + cur_artificial_var;
                    cur_artificial_var += 1;
                    basic_idx
                }

                RelOp::Ge => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);

                    coeffs = into_resized(coeffs, num_total_vars);
                    coeffs.append(num_vars + cur_slack_var, -1.0);
                    cur_slack_var += 1;

                    artificial_obj_val += bound;
                    artificial_multipliers.append(basic_vars.len(), 1.0);

                    let basic_idx = num_slack_vars + cur_artificial_var;
                    cur_artificial_var += 1;

                    basic_idx
                }
            };

            basic_vars_inv[num_vars + basic_idx] = basic_vars.len();
            basic_vars.push(num_vars + basic_idx);
            coeffs = into_resized(coeffs, num_total_vars);
            coeffs.append(num_vars + basic_idx, 1.0);
            orig_constraints = orig_constraints.append_outer_csvec(coeffs.view());
            orig_bounds.push(bound);
        }

        let orig_constraints_csc = orig_constraints.to_csc();

        let mut non_basic_vars = vec![];
        let mut non_basic_vars_inv = vec![SENTINEL; num_total_vars];
        let mut cur_obj = vec![];
        let mut non_basic_col_sq_norms = vec![];
        for var in 0..num_total_vars {
            if basic_vars_inv[var] == SENTINEL {
                non_basic_vars_inv[var] = non_basic_vars.len();
                non_basic_vars.push(var);

                let col = orig_constraints_csc.outer_view(var).unwrap();

                if num_artificial_vars > 0 {
                    cur_obj.push(artificial_multipliers.dot(&col));
                } else {
                    cur_obj.push(obj[var]);
                }

                if enable_steepest_edge {
                    non_basic_col_sq_norms.push(col.squared_l2_norm());
                }
            }
        }

        let cur_obj_val = if num_artificial_vars > 0 {
            artificial_obj_val
        } else {
            0.0
        };

        let mut scratch = ScratchSpace::with_capacity(num_constraints);
        let lu_factors = lu_factorize(
            basic_vars.len(),
            |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    .unwrap()
                    .into_raw_storage()
            },
            0.1,
            &mut scratch,
        )
        .unwrap();
        let lu_factors_transp = lu_factors.transpose();

        let cur_bounds = orig_bounds.clone();
        Tableau {
            num_vars,
            num_slack_vars,
            num_artificial_vars,
            orig_obj: obj,
            orig_constraints,
            orig_constraints_csc,
            orig_bounds,
            set_vars: HashMap::new(),
            enable_steepest_edge,
            basis_solver: BasisSolver {
                lu_factors,
                lu_factors_transp,
                scratch,
                eta_matrices: EtaMatrices::new(num_constraints),
                rhs: ScatteredVec::empty(num_constraints),
            },
            col_coeffs: SparseVec::new(),
            eta_matrix_coeffs: SparseVec::new(),
            sq_norms_update_helper: ScatteredVec::empty(num_total_vars - num_constraints),
            row_coeffs: ScatteredVec::empty(num_total_vars - num_constraints),
            basic_vars,
            basic_vars_inv,
            cur_bounds,
            non_basic_vars,
            non_basic_vars_inv,
            cur_obj,
            non_basic_col_sq_norms,
            cur_obj_val,
        }
    }

    pub fn num_total_vars(&self) -> usize {
        self.num_vars + self.num_slack_vars + self.num_artificial_vars
    }

    pub fn num_constraints(&self) -> usize {
        self.orig_constraints.rows()
    }

    pub fn cur_obj_val(&self) -> f64 {
        self.cur_obj_val
    }

    pub fn cur_solution(&self) -> Vec<f64> {
        let mut values = vec![0.0; self.num_vars];
        for (i, bv) in self.basic_vars.iter().enumerate() {
            if *bv < self.num_vars {
                values[*bv] = self.cur_bounds[i];
            }
        }
        for (&var, &val) in self.set_vars.iter() {
            assert_eq!(values[var], 0.0);
            values[var] = val;
        }
        values
    }

    pub fn move_to_solution(mut self, solution: &[f64]) -> Result<Self, Error> {
        assert_eq!(solution.len(), self.num_vars);

        // fill the slack variables.
        let mut values = Vec::with_capacity(self.num_vars + self.num_slack_vars);
        values.extend_from_slice(solution);
        values.resize(self.num_vars + self.num_slack_vars, 0.0);

        for (r, coeffs) in self.orig_constraints.outer_iterator().enumerate() {
            let bound = self.orig_bounds[r];

            let mut relevant_vars = CsVec::empty(coeffs.dim());
            for &i in coeffs.indices() {
                if i >= self.num_vars {
                    break;
                }

                relevant_vars.append(i, values[i]);
            }

            let mut accum = 0.0;
            for (i, coeff) in coeffs.iter() {
                if i < self.num_vars {
                    accum += coeff * values[i];
                } else if i < self.num_vars + self.num_slack_vars {
                    let mut val = (bound - accum) / coeff;
                    if f64::abs(val) < 1e-8 {
                        val = 0.0;
                    } else if val < 0.0 {
                        error!(
                            "Constraint #{} {:?}={} not satisfied, relevant vars: {:?}, slack var: {}",
                            r, coeffs, bound, relevant_vars, i
                        );
                        return Err(Error::Infeasible);
                    }
                    values[i] = val;
                    accum = bound;
                    break;
                } else {
                    break;
                }
            }

            if f64::abs(accum - bound) > 1e-8 {
                // necessary to check in case of equality constraint.
                error!(
                    "Equality constraint #{} {:?}={} not satisfied, relevant vars: {:?}",
                    r, coeffs, bound, relevant_vars
                );
                return Err(Error::Infeasible);
            }
        }

        let basic_vars = self.assign_basic_vars(&values)?;

        // Transition is feasible, we can remove artificial vars.
        self.remove_artificial_vars();
        self.set_vars.clear();
        self.reset_basis(basic_vars);
        Ok(self)
    }

    pub fn canonicalize(mut self) -> Result<Self, Error> {
        let mut cur_artificial_vars = self.num_artificial_vars;
        trace!("TAB {:?}", self);
        for iter in 0.. {
            if iter % 100 == 0 {
                debug!(
                    "canonicalize iter {}: art. objective: {}, art. vars: {}, nnz: {}",
                    iter,
                    self.cur_obj_val,
                    cur_artificial_vars,
                    self.nnz(),
                );
            }

            if cur_artificial_vars == 0 {
                debug!("canonicalized in {} iters, nnz: {}", iter + 1, self.nnz());
                break;
            }

            if let Some(c_entering) = self.choose_entering_col() {
                let entering_var = self.non_basic_vars[c_entering];
                self.calc_col_coeffs(c_entering);
                let (r_pivot, pivot_coeff) = self.choose_pivot_row()?;
                self.calc_row_coeffs(r_pivot);
                let leaving_var = self.pivot(c_entering, r_pivot, pivot_coeff);

                let art_vars_start = self.num_vars + self.num_slack_vars;
                match (entering_var < art_vars_start, leaving_var < art_vars_start) {
                    (true, false) => cur_artificial_vars -= 1,
                    (false, true) => cur_artificial_vars += 1,
                    _ => {}
                }
            } else {
                break;
            }
        }

        if self.cur_obj_val > 1e-8 {
            return Err(Error::Infeasible);
        }

        if cur_artificial_vars > 0 {
            panic!("{} artificial vars not eliminated!", cur_artificial_vars);
        }

        self.remove_artificial_vars();

        self.basic_vars_inv.truncate(self.num_total_vars());
        self.non_basic_vars_inv.truncate(self.num_total_vars());

        let mut new_non_basic_vars = vec![];
        let mut new_sq_norms = vec![];
        for (i, &var) in self.non_basic_vars.iter().enumerate() {
            if var < self.num_total_vars() {
                self.non_basic_vars_inv[var] = new_non_basic_vars.len();
                new_non_basic_vars.push(var);
                if self.enable_steepest_edge {
                    new_sq_norms.push(self.non_basic_col_sq_norms[i]);
                }
            }
        }
        self.non_basic_vars = new_non_basic_vars;
        self.non_basic_col_sq_norms = new_sq_norms;

        self.row_coeffs.clear_and_resize(self.non_basic_vars.len());
        self.sq_norms_update_helper
            .clear_and_resize(self.non_basic_vars.len());
        self.non_basic_col_sq_norms
            .truncate(self.non_basic_vars.len());

        self.recalc_cur_obj();

        trace!("CANONICAL TAB {:?}", self);
        Ok(self)
    }

    pub fn optimize(mut self) -> Result<Self, Error> {
        if self.num_artificial_vars > 0 {
            self = self.canonicalize()?;
        }

        for iter in 0.. {
            if iter % 100 == 0 {
                debug!(
                    "optimize iter {}: objective: {}, nnz: {}",
                    iter,
                    self.cur_obj_val(),
                    self.nnz()
                );
            }

            if let Some(c_entering) = self.choose_entering_col() {
                self.calc_col_coeffs(c_entering);
                let (r_pivot, pivot_coeff) = self.choose_pivot_row()?;
                self.calc_row_coeffs(r_pivot);
                self.pivot(c_entering, r_pivot, pivot_coeff);
            } else {
                debug!(
                    "found optimum: {} in {} iterations, nnz: {}",
                    self.cur_obj_val(),
                    iter + 1,
                    self.nnz(),
                );
                break;
            }
        }

        trace!("OPTIMIZED TAB {:?}", self);
        Ok(self)
    }

    pub fn restore_feasibility(mut self) -> Result<Self, Error> {
        if self.num_artificial_vars > 0 {
            self = self.canonicalize()?;
        }

        for iter in 0.. {
            if iter % 100 == 0 {
                debug!(
                    "restore feasibility iter {}: objective: {}, nnz: {}",
                    iter,
                    self.cur_obj_val(),
                    self.nnz(),
                );
            }

            let r_pivot = self.choose_pivot_row_dual();
            if self.cur_bounds[r_pivot] >= -1e-8 {
                debug!(
                    "restored feasibility in {} iterations, nnz: {}",
                    iter + 1,
                    self.nnz()
                );
                break;
            }

            self.calc_row_coeffs(r_pivot);
            let (c_entering, pivot_coeff) = self.choose_entering_col_dual()?;
            self.calc_col_coeffs(c_entering);
            self.pivot(c_entering, r_pivot, pivot_coeff);
        }

        Ok(self)
    }

    /// Precondition: optimality
    pub fn set_var(mut self, var: usize, val: f64) -> Result<Self, Error> {
        assert_eq!(self.num_artificial_vars, 0);
        assert!(self.set_vars.insert(var, val).is_none());

        let basic_row = self.basic_vars.iter().position(|&v| v == var);
        let non_basic_col = self.non_basic_vars.iter().position(|&v| v == var);

        if let Some(r) = basic_row {
            // if var was basic, remove it.
            self.cur_bounds[r] -= val;
            self.calc_row_coeffs(r);
            let (c_entering, pivot_coeff) = self.choose_entering_col_dual()?;
            self.calc_col_coeffs(c_entering);
            self.pivot(c_entering, r, pivot_coeff);
        } else if let Some(c) = non_basic_col {
            self.calc_col_coeffs(c);
            for (r, coeff) in self.col_coeffs.iter() {
                self.cur_bounds[r] -= val * coeff;
            }
            self.cur_obj_val -= val * self.cur_obj[c];
        } else {
            panic!(
                "couldn't find var {} in either basic or non-basic variables",
                var
            );
        }

        self = self.restore_feasibility()?;
        self = self.optimize().unwrap();
        Ok(self)
    }

    /// Precondition: optimality.
    /// Return true if the var was really unset.
    pub fn unset_var(mut self, var: usize) -> Result<(Self, bool), Error> {
        if let Some(val) = self.set_vars.remove(&var) {
            let col = self.non_basic_vars.iter().position(|&v| v == var).unwrap();
            self.calc_col_coeffs(col);
            for (r, coeff) in self.col_coeffs.iter() {
                self.cur_bounds[r] += val * coeff;
            }
            self.cur_obj_val += val * self.cur_obj[col];

            self = self.restore_feasibility()?;
            self = self.optimize().unwrap();
            Ok((self, true))
        } else {
            Ok((self, false))
        }
    }

    pub fn add_constraint(self, coeffs: CsVec, rel_op: RelOp, bound: f64) -> Result<Self, Error> {
        let (coeffs, bound) = match rel_op {
            RelOp::Le => (coeffs, bound),
            RelOp::Ge => (coeffs.map(|coeff| -coeff), -bound),
            RelOp::Eq => unimplemented!(),
        };
        assert_eq!(coeffs.dim(), self.num_vars);
        self.add_le_constraint_impl(coeffs, bound)
    }

    // TODO: remove_constraint

    fn add_le_constraint_impl(mut self, mut coeffs: CsVec, orig_bound: f64) -> Result<Self, Error> {
        assert_eq!(self.num_artificial_vars, 0);
        // TODO: assert optimality.

        // each <= constraint adds a slack var
        let new_num_total_vars = self.num_total_vars() + 1;
        let mut new_orig_constraints = CsMat::empty(CompressedStorage::CSR, new_num_total_vars);
        for row in self.orig_constraints.outer_iterator() {
            new_orig_constraints =
                new_orig_constraints.append_outer_csvec(resized_view(&row, new_num_total_vars));
        }

        let slack_var = self.num_vars + self.num_slack_vars;
        self.num_slack_vars += 1;

        self.orig_obj.push(0.0);

        coeffs = into_resized(coeffs, new_num_total_vars);
        coeffs.append(slack_var, 1.0);
        new_orig_constraints = new_orig_constraints.append_outer_csvec(coeffs.view());

        self.orig_bounds.push(orig_bound);

        self.basic_vars_inv.push(self.basic_vars.len());
        self.basic_vars.push(slack_var);
        self.non_basic_vars_inv.push(SENTINEL);

        self.orig_constraints = new_orig_constraints;
        self.orig_constraints_csc = self.orig_constraints.to_csc();

        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars);

        self.recalc_cur_bounds();

        if self.enable_steepest_edge {
            // existing tableau rows didn't change, so we calc the last row
            // and add its contribution to the sq. norms.
            self.calc_row_coeffs(self.num_constraints() - 1);
            for (c, &coeff) in self.row_coeffs.iter() {
                self.non_basic_col_sq_norms[c] += coeff * coeff;
            }
        }

        self = self.restore_feasibility()?;
        self = self.optimize().unwrap();
        Ok(self)
    }

    pub fn add_gomory_cut(mut self, var: usize) -> Result<Self, Error> {
        // TODO: assert optimality

        let basic_row = self.basic_vars_inv[var];
        if basic_row == SENTINEL {
            panic!("var {} is not basic!", var);
        }

        self.calc_row_coeffs(basic_row);

        let mut cut_coeffs = SparseVec::new();
        for (col, &coeff) in self.row_coeffs.iter() {
            let var = self.non_basic_vars[col];
            cut_coeffs.push(var, coeff.floor() - coeff);
        }

        let cut_bound = self.cur_bounds[basic_row].floor() - self.cur_bounds[basic_row];
        let num_total_vars = self.num_total_vars();
        self.add_le_constraint_impl(cut_coeffs.into_csvec(num_total_vars), cut_bound)
    }

    pub fn print_stats(&self) {
        info!(
            "tableau stats: num_vars={} num_slack_vars={} num_artificial_vars={} num_constraints={}, constraints nnz={}",
            self.num_vars,
            self.num_slack_vars,
            self.num_artificial_vars,
            self.orig_constraints.rows(),
            self.orig_constraints.nnz(),
        );
    }

    fn nnz(&self) -> usize {
        self.basis_solver.lu_factors.nnz()
    }

    /// Calculate current coeffs column for a single non-basic variable.
    fn calc_col_coeffs(&mut self, c_var: usize) {
        let var = self.non_basic_vars[c_var];
        let orig_col = self.orig_constraints_csc.outer_view(var).unwrap();
        self.basis_solver
            .solve(orig_col.iter())
            .to_sparse_vec(&mut self.col_coeffs);
    }

    fn calc_eta_matrix_coeffs(&mut self, r_leaving: usize, pivot_coeff: f64) {
        self.eta_matrix_coeffs.clear();
        for (r, &coeff) in self.col_coeffs.iter() {
            let val = if r == r_leaving {
                (coeff - 1.0) / pivot_coeff
            } else {
                coeff / pivot_coeff
            };
            self.eta_matrix_coeffs.push(r, val);
        }
    }

    /// Calculate current coeffs row for a single constraint (permuted according to non_basic_vars).
    fn calc_row_coeffs(&mut self, r_constr: usize) {
        let tmp = self
            .basis_solver
            .solve_transp(std::iter::once((r_constr, &1.0)));
        self.row_coeffs.clear_and_resize(self.non_basic_vars.len());
        for (r, &coeff) in tmp.iter() {
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                let idx = self.non_basic_vars_inv[v];
                if idx != SENTINEL {
                    *self.row_coeffs.get_mut(idx) += val * coeff;
                }
            }
        }
    }

    fn pivot(&mut self, c_entering: usize, r_leaving: usize, pivot_coeff: f64) -> usize {
        // TODO: periodically (say, every 1000 pivots) recalc cur_bounds and cur_obj
        // from scratch for numerical stability.

        let pivot_bound = self.cur_bounds[r_leaving] / pivot_coeff;
        for (r, coeff) in self.col_coeffs.iter() {
            if r == r_leaving {
                self.cur_bounds[r] = pivot_bound;
            } else {
                self.cur_bounds[r] -= pivot_bound * coeff;
            }
        }

        self.cur_obj_val -= self.cur_obj[c_entering] * pivot_bound;

        let pivot_obj = self.cur_obj[c_entering] / pivot_coeff;
        for (c, &coeff) in self.row_coeffs.iter() {
            if c == c_entering {
                self.cur_obj[c] = -pivot_obj;
            } else {
                self.cur_obj[c] -= pivot_obj * coeff;
            }
        }

        self.calc_eta_matrix_coeffs(r_leaving, pivot_coeff);

        if self.enable_steepest_edge {
            // Computations for the steepest edge pivoting rule. See
            // Vanderbei, Robert J. "Linear Programming: Foundations and Extensions." (2001).
            // p. 149.

            let tmp = self
                .basis_solver
                .solve_transp(self.eta_matrix_coeffs.iter());
            // now tmp contains the (w - v)/x_i vector.

            // Calculate transp(N) * (w - v) / x_1
            self.sq_norms_update_helper.clear();
            for (r, &coeff) in tmp.iter() {
                for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                    let idx = self.non_basic_vars_inv[v];
                    if idx != SENTINEL {
                        *self.sq_norms_update_helper.get_mut(idx) += val * coeff;
                    }
                }
            }

            let eta_sq_norm = self.eta_matrix_coeffs.sq_norm();
            for (c, &r_coeff) in self.row_coeffs.iter() {
                if c == c_entering {
                    self.non_basic_col_sq_norms[c] = eta_sq_norm - 1.0 + 2.0 / pivot_coeff;
                } else {
                    self.non_basic_col_sq_norms[c] +=
                        -2.0 * r_coeff * self.sq_norms_update_helper.get(c)
                            + eta_sq_norm * r_coeff * r_coeff;
                }
            }
        }

        let entering_var = self.non_basic_vars[c_entering];
        let leaving_var = std::mem::replace(&mut self.basic_vars[r_leaving], entering_var);
        std::mem::replace(&mut self.non_basic_vars[c_entering], leaving_var);
        self.basic_vars_inv[entering_var] = r_leaving;
        self.basic_vars_inv[leaving_var] = SENTINEL;
        self.non_basic_vars_inv[entering_var] = SENTINEL;
        self.non_basic_vars_inv[leaving_var] = c_entering;

        let eta_matrices_nnz = self.basis_solver.eta_matrices.coeff_cols.nnz();
        if eta_matrices_nnz < self.basis_solver.lu_factors.nnz() / 2 {
            self.basis_solver
                .push_eta_matrix(r_leaving, &self.eta_matrix_coeffs);
        } else {
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars);
        }

        trace!(
            "PIVOT entering {} (col #{}) leaving {} (row #{})",
            entering_var,
            c_entering,
            leaving_var,
            r_leaving,
        );

        leaving_var
    }

    // index in the non_basic_vars permutation.
    fn choose_entering_col(&self) -> Option<usize> {
        let mut entering = None;
        let mut entering_val = None;
        for c in 0..self.cur_obj.len() {
            let var = self.non_basic_vars[c];
            // set_vars.is_empty() check results in a small, but significant perf improvement.
            if self.cur_obj[c] < 1e-8
                || (!self.set_vars.is_empty() && self.set_vars.contains_key(&var))
            {
                continue;
            }

            let val = if self.num_artificial_vars == 0 && self.enable_steepest_edge {
                self.cur_obj[c] * self.cur_obj[c] / (self.non_basic_col_sq_norms[c] + 1.0)
            } else {
                // TODO: simple "biggest coeff" rule seems to perform much better than
                // the steepest edge rule for minimizing artificial objective (phase 1).
                // Why is that?
                self.cur_obj[c]
            };
            if entering_val.is_none() || val > entering_val.unwrap() {
                entering = Some(c);
                entering_val = Some(val);
            }
        }

        entering
    }

    /// returns index into basic_vars
    fn choose_pivot_row(&self) -> Result<(usize, f64), Error> {
        let mut leaving = 0;
        let mut leaving_val = None;
        let mut leaving_coeff = 0.0;
        for (r, &coeff) in self.col_coeffs.iter() {
            if coeff < 1e-8 {
                continue;
            }

            let cur_val = self.cur_bounds[r] / coeff;

            let should_choose = leaving_val.is_none()
                || cur_val < leaving_val.unwrap() - 1e-8
                || (cur_val < leaving_val.unwrap() + 1e-8
                    // There is uncertainty in choosing the leaving variable row.
                    // Choose the one with the biggest absolute coeff for the reasons of
                    // numerical stability. (NOTE: coeff is positive).
                    && (coeff > leaving_coeff + 1e-8
                        || coeff > leaving_coeff - 1e-8
                            // There is still uncertainty, choose based on the column index.
                            // NOTE: this still doesn't guarantee the absence of cycling.
                            && r < leaving));

            if should_choose {
                leaving = r;
                leaving_val = Some(cur_val);
                leaving_coeff = coeff;
            }
        }

        if leaving_val.is_none() {
            return Err(Error::Unbounded);
        }

        Ok((leaving, leaving_coeff))
    }

    fn choose_pivot_row_dual(&self) -> usize {
        let mut i_row = 0;
        let mut coeff = self.cur_bounds[0];
        for i in 1..self.num_constraints() {
            if self.cur_bounds[i] < coeff {
                i_row = i;
                coeff = self.cur_bounds[i];
            }
        }

        i_row
    }

    /// returns index into non_basic_vars
    fn choose_entering_col_dual(&self) -> Result<(usize, f64), Error> {
        let mut entering = 0;
        let mut entering_val = None;
        let mut entering_coeff = 0.0;
        for (c, &coeff) in self.row_coeffs.iter() {
            let var = self.non_basic_vars[c];
            // set_vars.is_empty() check results in a small, but significant perf improvement.
            if coeff > -1e-8 || (!self.set_vars.is_empty() && self.set_vars.contains_key(&var)) {
                continue;
            }

            let cur_val = self.cur_obj[c] / coeff; // obj[v] and row[v] are both negative

            // See comments in `choose_pivot_row`. NOTE: coeff is negative.
            let should_choose = entering_val.is_none()
                || cur_val < entering_val.unwrap() - 1e-8
                || (cur_val < entering_val.unwrap() + 1e-8
                    && (-coeff > -entering_coeff + 1e-8
                        || -coeff > -entering_coeff - 1e-8 && c < entering));

            if should_choose {
                entering = c;
                entering_val = Some(cur_val);
                entering_coeff = coeff;
            }
        }

        if entering_val.is_none() {
            return Err(Error::Infeasible);
        }

        Ok((entering, entering_coeff))
    }

    fn remove_artificial_vars(&mut self) {
        if self.num_artificial_vars == 0 {
            return;
        }

        self.num_artificial_vars = 0;
        self.orig_obj.truncate(self.num_total_vars());

        let mut new_constraints = CsMat::empty(CompressedStorage::CSR, self.num_total_vars());
        for row in self.orig_constraints.outer_iterator() {
            new_constraints =
                new_constraints.append_outer_csvec(resized_view(&row, self.num_total_vars()));
        }
        self.orig_constraints = new_constraints;
        self.orig_constraints_csc = self.orig_constraints.to_csc();
    }

    fn recalc_cur_bounds(&mut self) {
        let mut cur_bounds = self.orig_bounds.clone();
        for (&var, &val) in self.set_vars.iter() {
            for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
                cur_bounds[r] -= val * coeff;
            }
        }
        self.basis_solver
            .lu_factors
            .solve_dense(&mut cur_bounds, &mut self.basis_solver.scratch);
        self.cur_bounds = cur_bounds;
        for b in &mut self.cur_bounds {
            if f64::abs(*b) < 1e-8 {
                *b = 0.0;
            }
        }
    }

    fn recalc_cur_obj(&mut self) {
        if self.basis_solver.eta_matrices.len() > 0 {
            self.basis_solver
                .reset(&self.orig_constraints_csc, &self.basic_vars);
        }

        let multipliers = {
            let mut obj_coeffs = vec![0.0; self.num_constraints()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                obj_coeffs[c] = -self.orig_obj[var];
            }
            self.basis_solver
                .lu_factors_transp
                .solve_dense(&mut obj_coeffs, &mut self.basis_solver.scratch);
            ArrayVec::from(obj_coeffs)
        };

        self.cur_obj.clear();
        for &var in &self.non_basic_vars {
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let mut val = self.orig_obj[var] + col.dot(&multipliers);
            if f64::abs(val) < 1e-8 {
                val = 0.0;
            }
            self.cur_obj.push(val);
        }

        self.cur_obj_val = 0.0;
        for (c, &var) in self.basic_vars.iter().enumerate() {
            self.cur_obj_val += -self.orig_obj[var] * self.cur_bounds[c];
        }
        for (&var, &val) in self.set_vars.iter() {
            self.cur_obj_val += -self.orig_obj[var] * val;
        }
    }

    fn recalc_cur_sq_norms(&mut self) {
        self.non_basic_col_sq_norms.clear();
        for &var in &self.non_basic_vars {
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let sq_norm = self.basis_solver.solve(col.iter()).sq_norm();
            self.non_basic_col_sq_norms.push(sq_norm);
        }
    }

    fn reset_basis(&mut self, basic_vars: Vec<usize>) {
        assert_eq!(self.num_artificial_vars, 0);
        assert_eq!(basic_vars.len(), self.num_constraints());

        self.basic_vars = basic_vars;
        self.basic_vars_inv.clear();
        self.basic_vars_inv.resize(self.num_total_vars(), SENTINEL);
        for (i, &v) in self.basic_vars.iter().enumerate() {
            self.basic_vars_inv[v] = i;
        }

        self.non_basic_vars_inv.clear();
        self.non_basic_vars_inv
            .resize(self.num_total_vars(), SENTINEL);
        self.non_basic_vars.clear();
        for v in 0..self.num_total_vars() {
            if self.basic_vars_inv[v] == SENTINEL {
                self.non_basic_vars_inv[v] = self.non_basic_vars.len();
                self.non_basic_vars.push(v);
            }
        }

        self.basis_solver
            .reset(&self.orig_constraints_csc, &self.basic_vars);

        self.recalc_cur_bounds();
        self.recalc_cur_obj();
        if self.enable_steepest_edge {
            self.recalc_cur_sq_norms();
        }
    }

    /// for each constraint try to find a basic var using depth-first search.
    fn assign_basic_vars(&self, values: &[f64]) -> Result<Vec<usize>, Error> {
        assert_eq!(values.len(), self.num_vars + self.num_slack_vars);

        let mut is_var_used = vec![false; self.num_vars + self.num_slack_vars];
        let mut non_zero_vars_left = (0..values.len())
            .filter(|v| f64::abs(values[*v]) > 1e-8)
            .count();

        let (constr2vars, var2constrs) = {
            let mut c2vs = Vec::with_capacity(self.num_constraints());
            let mut v2cs = vec![vec![]; self.orig_constraints.cols()];
            for (c, row) in self.orig_constraints.outer_iterator().enumerate() {
                let mut step_vars: Vec<usize> = row
                    .iter()
                    .filter(|(v, coeff)| *v < values.len() && f64::abs(**coeff) > 1e-8)
                    .map(|(v, _)| v)
                    .collect();
                // try non-zero variables first.
                step_vars.sort_by_key(|v| (f64::abs(values[*v]) < 1e-8) as u32);
                for &v in &step_vars {
                    v2cs[v].push(c);
                }
                c2vs.push(step_vars);
            }
            (c2vs, v2cs)
        };

        #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug)]
        struct ConstraintInfo {
            is_used: bool,
            num_free_vars: usize,
            idx: usize,
        }

        let mut constr2info: Vec<ConstraintInfo> = constr2vars
            .iter()
            .enumerate()
            .map(|(idx, vs)| ConstraintInfo {
                is_used: false,
                num_free_vars: vs.len(),
                idx,
            })
            .collect();
        let mut sorted_constr_infos: BTreeSet<ConstraintInfo> =
            constr2info.iter().cloned().collect();

        struct Step<'a> {
            constraint: usize,
            vars: &'a [usize],
            cur_idx: usize,
            is_first: bool,
        }

        let new_step = |constr2info: &mut Vec<ConstraintInfo>,
                        sorted_constr_infos: &mut BTreeSet<ConstraintInfo>|
         -> Step {
            // choose constraint with the least number of free (undecided) vars.
            let mut info = sorted_constr_infos.iter().next().unwrap().clone();
            assert!(!info.is_used);
            sorted_constr_infos.take(&info).unwrap();
            info.is_used = true;
            constr2info[info.idx] = info.clone();
            sorted_constr_infos.insert(info.clone());

            Step {
                constraint: info.idx,
                vars: &constr2vars[info.idx],
                cur_idx: 0,
                is_first: true,
            }
        };

        let mut dfs_stack = vec![new_step(&mut constr2info, &mut sorted_constr_infos)];
        loop {
            let cur_constr = dfs_stack.len() - 1;
            let mut cur_step = dfs_stack.last_mut().unwrap();
            if cur_step.is_first {
                cur_step.is_first = false;
            } else {
                let var = cur_step.vars[cur_step.cur_idx];

                is_var_used[var] = false;
                for &c in &var2constrs[var] {
                    let mut info = constr2info[c].clone();
                    sorted_constr_infos.take(&info).unwrap();
                    info.num_free_vars += 1;
                    constr2info[info.idx] = info.clone();
                    sorted_constr_infos.insert(info.clone());
                }

                if f64::abs(values[var]) > 1e-8 {
                    non_zero_vars_left += 1;
                }
                cur_step.cur_idx += 1;
            }

            while cur_step.cur_idx < cur_step.vars.len()
                && is_var_used[cur_step.vars[cur_step.cur_idx]]
            {
                cur_step.cur_idx += 1;
            }

            if cur_step.cur_idx == cur_step.vars.len() {
                let mut info = constr2info[cur_step.constraint].clone();
                assert!(info.is_used);
                sorted_constr_infos.take(&info).unwrap();
                info.is_used = false;
                constr2info[info.idx] = info.clone();
                sorted_constr_infos.insert(info.clone());

                dfs_stack.pop();
                if dfs_stack.is_empty() {
                    error!("solution is not basic!");
                    return Err(Error::Infeasible);
                }
                continue;
            }

            let cur_var = cur_step.vars[cur_step.cur_idx];
            is_var_used[cur_var] = true;
            for &c in &var2constrs[cur_var] {
                let mut info = constr2info[c].clone();
                sorted_constr_infos.take(&info).unwrap();
                info.num_free_vars -= 1;
                constr2info[info.idx] = info.clone();
                sorted_constr_infos.insert(info.clone());
            }
            if f64::abs(values[cur_var]) > 1e-8 {
                non_zero_vars_left -= 1;
            }

            if non_zero_vars_left > self.num_constraints() - cur_constr - 1 {
                // Even if all the following basic vars are non-zero,
                // some non-zero vars will still be non-basic, which is prohibited.
                // Thus we need to backtrack.
                continue;
            }

            if dfs_stack.len() < self.num_constraints() {
                dfs_stack.push(new_step(&mut constr2info, &mut sorted_constr_infos));
            } else {
                break;
            }
        }

        let mut basic_vars = vec![0; self.num_constraints()];
        for step in &dfs_stack {
            basic_vars[step.constraint] = step.vars[step.cur_idx];
        }

        Ok(basic_vars)
    }
}

/// Stuff related to inversion of the basis matrix
#[derive(Clone)]
struct BasisSolver {
    lu_factors: LUFactors,
    lu_factors_transp: LUFactors,
    scratch: ScratchSpace,
    eta_matrices: EtaMatrices,
    rhs: ScatteredVec,
}

impl BasisSolver {
    fn push_eta_matrix(&mut self, r_leaving: usize, coeffs: &SparseVec) {
        self.eta_matrices.push(r_leaving, coeffs);
    }

    fn reset(&mut self, orig_constraints_csc: &CsMat, basic_vars: &[usize]) {
        self.scratch.clear_sparse(basic_vars.len());
        self.eta_matrices.clear_and_resize(basic_vars.len());
        self.rhs.clear_and_resize(basic_vars.len());
        self.lu_factors = lu_factorize(
            basic_vars.len(),
            |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    .unwrap()
                    .into_raw_storage()
            },
            0.1,
            &mut self.scratch,
        )
        .unwrap(); // TODO: When is singular basis matrix possible? Report as a proper error.
        self.lu_factors_transp = self.lu_factors.transpose();
    }

    fn solve<'a>(&mut self, rhs: impl Iterator<Item = (usize, &'a f64)>) -> &ScatteredVec {
        self.rhs.set(rhs);
        self.lu_factors.solve(&mut self.rhs, &mut self.scratch);

        // apply eta matrices (Vanderbei p.139)
        for idx in 0..self.eta_matrices.len() {
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            let coeff = *self.rhs.get(r_leaving);
            for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                *self.rhs.get_mut(r) -= coeff * val;
            }
        }

        &mut self.rhs
    }

    /// Pass right-hand side via self.rhs
    fn solve_transp<'a>(&mut self, rhs: impl Iterator<Item = (usize, &'a f64)>) -> &ScatteredVec {
        self.rhs.set(rhs);
        // apply eta matrices in reverse (Vanderbei p.139)
        for idx in (0..self.eta_matrices.len()).rev() {
            let mut coeff = 0.0;
            // eta col `dot` rhs_transp
            for (i, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                coeff += val * self.rhs.get(i);
            }
            let r_leaving = self.eta_matrices.leaving_rows[idx];
            *self.rhs.get_mut(r_leaving) -= coeff;
        }

        self.lu_factors_transp
            .solve(&mut self.rhs, &mut self.scratch);
        &mut self.rhs
    }
}

#[derive(Clone, Debug)]
struct EtaMatrices {
    leaving_rows: Vec<usize>,
    coeff_cols: SparseMat,
}

impl EtaMatrices {
    fn new(n_rows: usize) -> EtaMatrices {
        EtaMatrices {
            leaving_rows: vec![],
            coeff_cols: SparseMat::new(n_rows),
        }
    }

    fn len(&self) -> usize {
        self.leaving_rows.len()
    }

    fn clear_and_resize(&mut self, n_rows: usize) {
        self.leaving_rows.clear();
        self.coeff_cols.clear_and_resize(n_rows);
    }

    fn push(&mut self, leaving_row: usize, coeffs: &SparseVec) {
        self.leaving_rows.push(leaving_row);
        self.coeff_cols.append_col(coeffs.iter());
    }
}

fn into_resized(vec: CsVec, len: usize) -> CsVec {
    let (mut indices, mut data) = vec.into_raw_storage();

    while let Some(&i) = indices.last() {
        if i < len {
            // TODO: binary search
            break;
        }

        indices.pop();
        data.pop();
    }

    CsVec::new(len, indices, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use helpers::{assert_matrix_eq, to_sparse};

    #[test]
    fn initialize() {
        let tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 4.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Ge, 2.0),
                (to_sparse(&[0.0, 1.0]), RelOp::Eq, 3.0),
            ],
        );

        assert_eq!(tab.num_vars, 2);
        assert_eq!(tab.num_slack_vars, 2);
        assert_eq!(tab.num_artificial_vars, 2);

        assert_eq!(tab.orig_obj, vec![-2.0, -1.0, 0.0, 0.0, 0.0, 0.0]);

        let orig_constraints_ref = vec![
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, -1.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        ];
        assert_matrix_eq(&tab.orig_constraints, &orig_constraints_ref);

        assert_eq!(tab.basic_vars, vec![2, 4, 5]);
        assert_eq!(tab.cur_bounds, vec![4.0, 2.0, 3.0]);

        assert_eq!(tab.non_basic_vars, vec![0, 1, 3]);
        assert_eq!(tab.cur_obj, vec![1.0, 2.0, -1.0]);

        assert_eq!(tab.cur_obj_val, 5.0);

        assert_eq!(tab.basic_vars, vec![2, 4, 5]);
    }

    #[test]
    fn canonicalize() {
        let tab = Tableau::new(
            vec![-3.0, -4.0],
            vec![
                (to_sparse(&[1.0, 0.0]), RelOp::Ge, 10.0),
                (to_sparse(&[0.0, 1.0]), RelOp::Ge, 5.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 20.0),
                (to_sparse(&[-1.0, 4.0]), RelOp::Le, 20.0),
            ],
        )
        .canonicalize()
        .unwrap();

        assert_eq!(tab.num_vars, 2);
        assert_eq!(tab.num_slack_vars, 4);
        assert_eq!(tab.num_artificial_vars, 0);

        assert_eq!(tab.cur_bounds, vec![10.0, 5.0, 5.0, 10.0]);
        assert_eq!(tab.non_basic_vars, vec![2, 3]);
        assert_eq!(tab.cur_obj, vec![3.0, 4.0]);
        assert_eq!(tab.cur_obj_val, -50.0);

        assert_eq!(tab.basic_vars, vec![0, 1, 4, 5]);

        let infeasible = Tableau::new(
            vec![1.0, 1.0],
            vec![
                (to_sparse(&[1.0, 1.0]), RelOp::Ge, 10.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 5.0),
            ],
        )
        .canonicalize();
        assert_eq!(infeasible.unwrap_err(), Error::Infeasible);
    }

    #[test]
    fn optimize() {
        let mut tab = Tableau::new(
            vec![-3.0, -4.0],
            vec![
                (to_sparse(&[1.0, 0.0]), RelOp::Ge, 10.0),
                (to_sparse(&[0.0, 1.0]), RelOp::Ge, 5.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 20.0),
                (to_sparse(&[-1.0, 4.0]), RelOp::Le, 20.0),
            ],
        );

        tab = tab.optimize().unwrap();
        assert_eq!(tab.cur_solution(), vec![12.0, 8.0]);
        assert_eq!(tab.cur_obj_val(), -68.0);
    }

    #[test]
    fn move_to_solution() {
        let mut tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 4.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Ge, 2.0),
                (to_sparse(&[0.0, 1.0]), RelOp::Eq, 3.0),
            ],
        );

        assert_eq!(
            tab.clone().move_to_solution(&[0.0, 4.0]).err(),
            Some(Error::Infeasible)
        );
        assert_eq!(
            tab.clone().move_to_solution(&[0.5, 3.0]).err(),
            Some(Error::Infeasible)
        );

        tab = tab.move_to_solution(&[1.0, 3.0]).unwrap();
        assert_eq!(tab.cur_obj_val(), 5.0);

        tab = tab.optimize().unwrap();
        assert_eq!(tab.cur_solution(), vec![0.0, 3.0]);
        assert_eq!(tab.cur_obj_val(), 3.0);
    }

    #[test]
    fn set_unset_var() {
        let orig_tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 4.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Ge, 2.0),
            ],
        )
        .optimize()
        .unwrap();

        {
            let mut tab = orig_tab.clone().set_var(0, 3.0).unwrap();
            assert_eq!(tab.cur_solution(), vec![3.0, 0.0]);
            assert_eq!(tab.cur_obj_val(), 6.0);

            tab = tab.unset_var(0).unwrap().0;
            assert_eq!(tab.cur_solution(), vec![0.0, 2.0]);
            assert_eq!(tab.cur_obj_val(), 2.0);
        }

        {
            let mut tab = orig_tab.clone().set_var(1, 3.0).unwrap();
            assert_eq!(tab.cur_solution(), vec![0.0, 3.0]);
            assert_eq!(tab.cur_obj_val(), 3.0);

            tab = tab.unset_var(1).unwrap().0;
            assert_eq!(tab.cur_solution(), vec![0.0, 2.0]);
            assert_eq!(tab.cur_obj_val(), 2.0);
        }
    }

    #[test]
    fn add_constraint() {
        let orig_tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                (to_sparse(&[1.0, 1.0]), RelOp::Le, 4.0),
                (to_sparse(&[1.0, 1.0]), RelOp::Ge, 2.0),
            ],
        )
        .optimize()
        .unwrap();

        {
            let tab = orig_tab
                .clone()
                .add_constraint(to_sparse(&[-1.0, 1.0]), RelOp::Le, 0.0)
                .unwrap();
            assert_eq!(tab.cur_solution(), [1.0, 1.0]);
            assert_eq!(tab.cur_obj_val(), 3.0);
        }

        {
            let tab = orig_tab
                .clone()
                .set_var(1, 1.5)
                .unwrap()
                .add_constraint(to_sparse(&[-1.0, 1.0]), RelOp::Le, 0.0)
                .unwrap();
            assert_eq!(tab.cur_solution(), [1.5, 1.5]);
            assert_eq!(tab.cur_obj_val(), 4.5);
        }

        {
            let tab = orig_tab
                .clone()
                .add_constraint(to_sparse(&[-1.0, 1.0]), RelOp::Ge, 3.0)
                .unwrap();
            assert_eq!(tab.cur_solution(), [0.0, 3.0]);
            assert_eq!(tab.cur_obj_val(), 3.0);
        }
    }

    #[test]
    fn gomory_cut() {
        let mut tab = Tableau::new(
            vec![0.0, -1.0],
            vec![
                (to_sparse(&[3.0, 2.0]), RelOp::Le, 6.0),
                (to_sparse(&[-3.0, 2.0]), RelOp::Le, 0.0),
            ],
        )
        .optimize()
        .unwrap();

        assert_eq!(tab.cur_solution(), [1.0, 1.5]);
        assert_eq!(tab.cur_obj_val(), -1.5);

        tab = tab.add_gomory_cut(1).unwrap();
        let solution = tab.cur_solution();
        assert!(f64::abs(solution[0] - 2.0 / 3.0) < 1e-8);
        assert_eq!(solution[1], 1.0);
        assert_eq!(tab.cur_obj_val(), -1.0);

        tab = tab.add_gomory_cut(0).unwrap();
        let solution = tab.cur_solution();
        assert!(f64::abs(solution[0] - 1.0) < 1e-8);
        assert_eq!(solution[1], 1.0);
        assert_eq!(tab.cur_obj_val(), -1.0);
    }
}
