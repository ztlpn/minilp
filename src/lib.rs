#[macro_use]
extern crate log;

pub use sprs::CsVec;

use sprs::{CompressedStorage, CsMat};
use std::collections::{BTreeSet, HashMap};

mod lu;
use lu::{lu_factorize, LUFactors, ScatteredVec, ScratchSpace};

mod helpers;
use helpers::{resized_view, to_dense};

type ArrayVec = ndarray::Array1<f64>;

const SENTINEL: usize = 0usize.wrapping_sub(1);

#[derive(Clone)]
pub struct Tableau {
    num_vars: usize,
    num_slack_vars: usize,
    num_artificial_vars: usize,

    orig_obj: Vec<f64>,           // with negated coeffs
    orig_constraints: CsMat<f64>, // excluding bounds
    orig_constraints_csc: CsMat<f64>,
    orig_bounds: Vec<f64>,

    // LU factors of the basis matrix
    lu_factors: LUFactors,
    eta_matrices: Vec<EtaMatrix>,
    rhs: ScatteredVec,
    row_coeffs: ScatteredVec,
    scratch: ScratchSpace,

    basic_vars: Vec<usize>, // for each constraint the corresponding basic var.
    cur_bounds: Vec<f64>,

    non_basic_vars: Vec<usize>,     // remaining variables. (idx -> var)
    non_basic_vars_inv: Vec<usize>, // (var -> idx)
    cur_obj: Vec<f64>,

    cur_obj_val: f64,

    set_vars: HashMap<usize, f64>,
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
        write!(f, "lu_factors:\n{:?}", self.lu_factors)?;
        write!(f, "basic_vars:\n{:?}\n", self.basic_vars)?;
        write!(f, "cur_bounds:\n{:?}\n", self.cur_bounds)?;
        write!(f, "non_basic_vars:\n{:?}\n", self.non_basic_vars)?;
        write!(f, "non_basic_vars_inv:\n{:?}\n", self.non_basic_vars_inv)?;
        write!(f, "cur_obj:\n{:?}\n", self.cur_obj)?;
        write!(f, "cur_obj_val: {:?}\n", self.cur_obj_val)?;
        write!(f, "set_vars:\n{:?}\n", self.set_vars)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Constraint {
    Eq(CsVec<f64>, f64),
    Le(CsVec<f64>, f64),
    Ge(CsVec<f64>, f64),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Error {
    Infeasible,
    Unbounded,
}

impl Tableau {
    pub fn new(mut obj: Vec<f64>, constraints: Vec<Constraint>) -> Tableau {
        let num_vars = obj.len();
        let (num_slack_vars, num_artificial_vars) = {
            let mut s = 0;
            let mut a = 0;
            for constr in &constraints {
                match constr {
                    Constraint::Le(..) => s += 1,
                    Constraint::Eq(..) => a += 1,
                    Constraint::Ge(..) => {
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
        let mut is_basic_var = vec![false; num_total_vars];

        for constr in constraints.into_iter() {
            let (mut coeffs, bound, basic_idx) = match constr {
                Constraint::Le(coeffs, bound) => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);
                    let basic_idx = cur_slack_var;
                    cur_slack_var += 1;
                    (coeffs, bound, basic_idx)
                }

                Constraint::Eq(coeffs, bound) => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);
                    artificial_obj_val += bound;
                    artificial_multipliers.append(basic_vars.len(), 1.0);
                    let basic_idx = num_slack_vars + cur_artificial_var;
                    cur_artificial_var += 1;
                    (coeffs, bound, basic_idx)
                }

                Constraint::Ge(mut coeffs, bound) => {
                    assert_eq!(coeffs.dim(), num_vars);
                    assert!(bound >= 0.0);

                    coeffs = into_resized(coeffs, num_total_vars);
                    coeffs.append(num_vars + cur_slack_var, -1.0);
                    cur_slack_var += 1;

                    artificial_obj_val += bound;
                    artificial_multipliers.append(basic_vars.len(), 1.0);

                    let basic_idx = num_slack_vars + cur_artificial_var;
                    cur_artificial_var += 1;

                    (coeffs, bound, basic_idx)
                }
            };

            basic_vars.push(num_vars + basic_idx);
            is_basic_var[num_vars + basic_idx] = true;
            coeffs = into_resized(coeffs, num_total_vars);
            coeffs.append(num_vars + basic_idx, 1.0);
            orig_constraints = orig_constraints.append_outer_csvec(coeffs.view());
            orig_bounds.push(bound);
        }

        let (non_basic_vars, non_basic_vars_inv) = {
            let mut forward = vec![];
            let mut inv = vec![SENTINEL; num_total_vars];

            for v in 0..num_total_vars {
                if !is_basic_var[v] {
                    inv[v] = forward.len();
                    forward.push(v);
                }
            }
            (forward, inv)
        };

        let orig_constraints_csc = orig_constraints.to_csc();

        let cur_obj = {
            let mut res = vec![];
            for &var in &non_basic_vars {
                let col = orig_constraints_csc.outer_view(var).unwrap();
                res.push(artificial_multipliers.dot(&col));
            }
            res
        };

        let mut scratch = ScratchSpace::with_capacity(num_constraints);
        let lu_factors = lu_factorize(&orig_constraints_csc, &basic_vars, 0.1, &mut scratch);

        let cur_bounds = orig_bounds.clone();
        Tableau {
            num_vars,
            num_slack_vars,
            num_artificial_vars,
            orig_obj: obj,
            orig_constraints,
            orig_constraints_csc,
            orig_bounds,
            lu_factors,
            eta_matrices: vec![],
            rhs: ScatteredVec::empty(num_constraints),
            row_coeffs: ScatteredVec::empty(num_total_vars - num_constraints),
            scratch,
            basic_vars,
            cur_bounds,
            non_basic_vars,
            non_basic_vars_inv,
            cur_obj,
            cur_obj_val: artificial_obj_val,
            set_vars: HashMap::new(),
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

    pub fn move_to_solution(&mut self, solution: &[f64]) -> Result<(), Error> {
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

        self.basic_vars = basic_vars;
        self.set_vars.clear();

        // Transition is feasible, we can remove artificial vars.
        self.remove_artificial_vars();
        self.recalc_cur_state();
        Ok(())
    }

    pub fn canonicalize(&mut self) -> Result<(), Error> {
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
                let entering_coeffs = self.calc_col_coeffs(c_entering);
                let r_pivot = self.choose_pivot_row(&entering_coeffs)?;
                let row_coeffs = self.calc_row_coeffs(r_pivot);
                let leaving_var = self.pivot(c_entering, &entering_coeffs, r_pivot, &row_coeffs);

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

        if f64::abs(self.cur_obj_val) > 1e-8 {
            return Err(Error::Infeasible);
        }

        if cur_artificial_vars > 0 {
            panic!("{} artificial vars not eliminated!", cur_artificial_vars);
        }

        self.remove_artificial_vars();
        self.recalc_cur_state();
        trace!("CANONICAL TAB {:?}", self);
        Ok(())
    }

    pub fn optimize(&mut self) -> Result<(), Error> {
        if self.num_artificial_vars > 0 {
            self.canonicalize()?;
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
                let entering_coeffs = self.calc_col_coeffs(c_entering);
                let r_pivot = self.choose_pivot_row(&entering_coeffs)?;
                let row_coeffs = self.calc_row_coeffs(r_pivot);
                self.pivot(c_entering, &entering_coeffs, r_pivot, &row_coeffs);
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

        Ok(())
    }

    pub fn restore_feasibility(&mut self) -> Result<(), Error> {
        if self.num_artificial_vars > 0 {
            self.canonicalize()?;
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

            let row_coeffs = self.calc_row_coeffs(r_pivot);
            let c_entering = self.choose_entering_col_dual(&row_coeffs)?;
            let entering_coeffs = self.calc_col_coeffs(c_entering);
            self.pivot(c_entering, &entering_coeffs, r_pivot, &row_coeffs);
        }

        Ok(())
    }

    /// Precondition: optimality
    pub fn set_var(&mut self, var: usize, val: f64) -> Result<(), Error> {
        assert_eq!(self.num_artificial_vars, 0);
        assert!(self.set_vars.insert(var, val).is_none());
        // If additional constraint renders the problem infeasible, tableau can become garbled.
        // We protect against it by saving basic variables and returning to them if that happens.
        let orig_basic_vars = self.basic_vars.clone();
        if let Err(e) = self.set_var_impl(var, val) {
            self.set_vars.remove(&var);
            self.basic_vars = orig_basic_vars;
            self.recalc_cur_state();
            return Err(e);
        }
        Ok(())
    }

    fn set_var_impl(&mut self, var: usize, val: f64) -> Result<(), Error> {
        let basic_row = self.basic_vars.iter().position(|&v| v == var);
        let non_basic_col = self.non_basic_vars.iter().position(|&v| v == var);

        if let Some(r) = basic_row {
            // if var was basic, remove it.
            let row_coeffs = self.calc_row_coeffs(r);
            let c_entering = self.choose_entering_col_dual(&row_coeffs)?;
            let entering_coeffs = self.calc_col_coeffs(c_entering);
            self.cur_bounds[r] -= val;

            self.pivot(c_entering, &entering_coeffs, r, &row_coeffs);
        } else if let Some(c) = non_basic_col {
            let coeffs = self.calc_col_coeffs(c);
            for (r, coeff) in coeffs.iter() {
                self.cur_bounds[r] -= val * coeff;
            }
            self.cur_obj_val -= val * self.cur_obj[c];
        } else {
            panic!(
                "couldn't find var {} in either basic or non-basic variables",
                var
            );
        }

        self.restore_feasibility()?;
        self.optimize().unwrap();
        Ok(())
    }

    /// Precondition: optimality.
    /// Return true if the var was really unset.
    pub fn unset_var(&mut self, var: usize) -> Result<bool, Error> {
        if let Some(val) = self.set_vars.remove(&var) {
            let col = self.non_basic_vars.iter().position(|&v| v == var).unwrap();
            let coeffs = self.calc_col_coeffs(col);
            for (r, coeff) in coeffs.iter() {
                self.cur_bounds[r] += val * coeff;
            }
            self.cur_obj_val += val * self.cur_obj[col];

            self.restore_feasibility()?;
            self.optimize().unwrap();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub fn add_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        assert_eq!(self.num_artificial_vars, 0);
        // TODO: assert optimality.

        let mut tmp = self.clone();

        // each constraint adds a slack var
        let new_num_total_vars = tmp.num_total_vars() + constraints.len();
        let mut new_orig_constraints = CsMat::empty(CompressedStorage::CSR, new_num_total_vars);
        for row in tmp.orig_constraints.outer_iterator() {
            new_orig_constraints =
                new_orig_constraints.append_outer_csvec(resized_view(&row, new_num_total_vars));
        }

        for constr in constraints {
            let (mut coeffs, bound) = if let Constraint::Le(coeffs, bound) = constr {
                (coeffs.clone(), *bound)
            } else {
                unimplemented!();
            };

            assert_eq!(coeffs.dim(), self.num_vars);
            assert!(bound >= 0.0);

            let slack_var = tmp.num_vars + tmp.num_slack_vars;
            tmp.num_slack_vars += 1;

            tmp.orig_obj.push(0.0);

            coeffs = into_resized(coeffs, new_num_total_vars);
            coeffs.append(slack_var, 1.0);
            new_orig_constraints = new_orig_constraints.append_outer_csvec(coeffs.view());

            tmp.orig_bounds.push(bound);

            tmp.basic_vars.push(slack_var);
        }

        tmp.orig_constraints = new_orig_constraints;
        tmp.orig_constraints_csc = tmp.orig_constraints.to_csc();

        tmp.recalc_cur_state();

        tmp.restore_feasibility()?;
        tmp.optimize().unwrap();

        *self = tmp;
        Ok(())
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
        self.lu_factors.nnz()
    }

    fn update_solver(&mut self, r_leaving: usize, pivot_coeff: f64, entering_coeffs: &CsVec<f64>) {
        let eta_matrices_nnz = self
            .eta_matrices
            .iter()
            .map(|m| m.entering_coeffs.nnz())
            .sum::<usize>();
        if eta_matrices_nnz < self.lu_factors.nnz() / 2 {
            self.eta_matrices.push(EtaMatrix {
                r_leaving,
                pivot_coeff,
                entering_coeffs: entering_coeffs.clone(),
            });
        } else {
            self.eta_matrices.clear();

            self.lu_factors = lu_factorize(
                &self.orig_constraints_csc,
                &self.basic_vars,
                0.1,
                &mut self.scratch,
            );
        }
    }

    /// Calculate current coeffs column for a single non-basic variable.
    fn calc_col_coeffs(&mut self, c_var: usize) -> CsVec<f64> {
        let var = self.non_basic_vars[c_var];
        let orig_coeffs = self.orig_constraints_csc.outer_view(var).unwrap();
        self.rhs.set(orig_coeffs);

        self.lu_factors.solve(&mut self.rhs, &mut self.scratch);

        // apply eta matrices (Vanderbei p.139)
        for eta in &self.eta_matrices {
            let coeff = self.rhs.get(eta.r_leaving) / eta.pivot_coeff;
            self.rhs.mul_add(-coeff, eta.entering_coeffs.view());
            *self.rhs.get_mut(eta.r_leaving) += coeff;
        }

        let res = self.rhs.to_csvec();
        trace!("VAR {} COEFFS: {:?}", var, to_dense(&res));
        res
    }

    /// Calculate current coeffs row for a single constraint (permuted according to non_basic_vars).
    fn calc_row_coeffs(&mut self, r_constr: usize) -> CsVec<f64> {
        self.rhs.clear();
        *self.rhs.get_mut(r_constr) = 1.0;

        // apply eta matrices in reverse (Vanderbei p.139)
        for eta in self.eta_matrices.iter().rev() {
            let mut coeff = 0.0;
            // eta.2 `dot` rhs_transp
            for (i, &val) in eta.entering_coeffs.iter() {
                coeff += val * self.rhs.get(i);
            }
            coeff -= self.rhs.get(eta.r_leaving);
            coeff /= eta.pivot_coeff;

            *self.rhs.get_mut(eta.r_leaving) -= coeff;
        }

        self.lu_factors
            .solve_transp(&mut self.rhs, &mut self.scratch);

        self.row_coeffs.clear();
        for (r, &coeff) in self.rhs.to_csvec().iter() {
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                let idx = self.non_basic_vars_inv[v];
                if idx != SENTINEL {
                    *self.row_coeffs.get_mut(idx) += val * coeff;
                }
            }
        }

        let res = self.row_coeffs.to_csvec();
        trace!("CONSTR {} COEFFS {:?}", r_constr, to_dense(&res));
        res
    }

    fn pivot(
        &mut self,
        c_entering: usize,
        entering_coeffs: &CsVec<f64>,
        r_leaving: usize,
        row_coeffs: &CsVec<f64>,
    ) -> usize {
        let entering_var = self.non_basic_vars[c_entering];
        let leaving_var = std::mem::replace(&mut self.basic_vars[r_leaving], entering_var);
        std::mem::replace(&mut self.non_basic_vars[c_entering], leaving_var);
        self.non_basic_vars_inv[entering_var] = SENTINEL;
        self.non_basic_vars_inv[leaving_var] = c_entering;
        trace!(
            "PIVOT entering {} (col #{}) leaving {} (row #{})",
            entering_var,
            c_entering,
            leaving_var,
            r_leaving,
        );

        // TODO: save NNZ index
        let pivot_coeff = *entering_coeffs.get(r_leaving).unwrap();
        self.update_solver(r_leaving, pivot_coeff, &entering_coeffs);

        let pivot_bound = self.cur_bounds[r_leaving] / pivot_coeff;
        for (r, coeff) in entering_coeffs.iter() {
            if r == r_leaving {
                self.cur_bounds[r] = pivot_bound;
            } else {
                self.cur_bounds[r] -= pivot_bound * coeff;
            }

            if f64::abs(self.cur_bounds[r]) < 1e-8 {
                self.cur_bounds[r] = 0.0;
            }
        }

        self.cur_obj_val -= self.cur_obj[c_entering] * pivot_bound;

        let pivot_obj = self.cur_obj[c_entering] / pivot_coeff;
        for (c, coeff) in row_coeffs.iter() {
            if c == c_entering {
                self.cur_obj[c] = -pivot_obj;
            } else {
                self.cur_obj[c] -= pivot_obj * coeff;
            }

            if f64::abs(self.cur_obj[c]) < 1e-8 {
                self.cur_obj[c] = 0.0;
            }
        }

        leaving_var
    }

    // index in the non_basic_vars permutation.
    fn choose_entering_col(&self) -> Option<usize> {
        let mut entering = None;
        let mut entering_coeff = None;
        for c in 0..self.cur_obj.len() {
            let var = self.non_basic_vars[c];
            if self.cur_obj[c] < 1e-8 || self.set_vars.contains_key(&var) {
                continue;
            }

            if entering_coeff.is_none() || self.cur_obj[c] > entering_coeff.unwrap() {
                entering = Some(c);
                entering_coeff = Some(self.cur_obj[c]);
            }
        }

        entering
    }

    fn choose_pivot_row(&self, coeffs: &CsVec<f64>) -> Result<usize, Error> {
        let mut i_row = 0;
        let mut coeff = None;
        for (i, &c) in coeffs.iter() {
            if c < 1e-8 {
                continue;
            }

            let cur_coeff = self.cur_bounds[i] / c;
            if coeff.is_none() || coeff.unwrap() > cur_coeff {
                i_row = i;
                coeff = Some(cur_coeff);
            }
        }

        if coeff.is_none() {
            return Err(Error::Unbounded);
        }

        Ok(i_row)
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

    // index in the non_basic_vars permutation
    fn choose_entering_col_dual(&self, row_coeffs: &CsVec<f64>) -> Result<usize, Error> {
        // eprintln!("ROW {} CUR COEFFS {:?}\n", i_row, to_dense(&row));

        let mut entering = 0;
        let mut entering_val = None;
        for (c, &coeff) in row_coeffs.iter() {
            let var = self.non_basic_vars[c];
            if coeff > -1e-8 || self.set_vars.contains_key(&var) {
                continue;
            }

            let cur_val = self.cur_obj[c] / coeff; // obj[v] and row[v] are both negative
            if entering_val.is_none() || entering_val.unwrap() > cur_val {
                entering = c;
                entering_val = Some(cur_val);
            }
        }

        if entering_val.is_none() {
            return Err(Error::Infeasible);
        }

        Ok(entering)
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

    fn recalc_cur_state(&mut self) {
        assert_eq!(self.num_artificial_vars, 0);

        self.rhs.clear_and_resize(self.basic_vars.len());

        self.non_basic_vars_inv.clear();
        self.non_basic_vars_inv.resize(self.num_total_vars(), 0);
        for &v in &self.basic_vars {
            self.non_basic_vars_inv[v] = SENTINEL;
        }

        self.non_basic_vars.clear();
        for v in 0..self.num_total_vars() {
            if self.non_basic_vars_inv[v] != SENTINEL {
                self.non_basic_vars_inv[v] = self.non_basic_vars.len();
                self.non_basic_vars.push(v);
            }
        }

        self.row_coeffs.clear_and_resize(self.non_basic_vars.len());

        self.eta_matrices.clear();
        self.lu_factors = lu_factorize(
            &self.orig_constraints_csc,
            &self.basic_vars,
            0.1,
            &mut self.scratch,
        );

        let mut cur_bounds = self.orig_bounds.clone();
        for (&var, &val) in self.set_vars.iter() {
            for (r, &coeff) in self.orig_constraints_csc.outer_view(var).unwrap().iter() {
                cur_bounds[r] -= val * coeff;
            }
        }
        self.lu_factors.solve_dense(&mut cur_bounds);
        self.cur_bounds = cur_bounds;
        for b in &mut self.cur_bounds {
            if f64::abs(*b) < 1e-8 {
                *b = 0.0;
            }
        }

        self.cur_obj_val = 0.0;
        for (c, &var) in self.basic_vars.iter().enumerate() {
            self.cur_obj_val += -self.orig_obj[var] * self.cur_bounds[c];
        }
        for (&var, &val) in self.set_vars.iter() {
            self.cur_obj_val += -self.orig_obj[var] * val;
        }

        let multipliers = {
            let mut obj_coeffs = vec![0.0; self.num_constraints()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                obj_coeffs[c] = -self.orig_obj[var];
            }
            self.lu_factors.solve_dense_transp(&mut obj_coeffs);
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

#[derive(Clone, Debug)]
struct EtaMatrix {
    r_leaving: usize,
    pivot_coeff: f64,
    entering_coeffs: CsVec<f64>,
}

fn into_resized(vec: CsVec<f64>, len: usize) -> CsVec<f64> {
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
    use helpers::{assert_matrix_eq, to_sparse};
    use super::*;

    #[test]
    fn initialize() {
        let tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                Constraint::Le(to_sparse(&[1.0, 1.0]), 4.0),
                Constraint::Ge(to_sparse(&[1.0, 1.0]), 2.0),
                Constraint::Eq(to_sparse(&[0.0, 1.0]), 3.0),
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
        let mut tab = Tableau::new(
            vec![-3.0, -4.0],
            vec![
                Constraint::Ge(to_sparse(&[1.0, 0.0]), 10.0),
                Constraint::Ge(to_sparse(&[0.0, 1.0]), 5.0),
                Constraint::Le(to_sparse(&[1.0, 1.0]), 20.0),
                Constraint::Le(to_sparse(&[-1.0, 4.0]), 20.0),
            ],
        );

        tab.canonicalize().unwrap();

        assert_eq!(tab.num_vars, 2);
        assert_eq!(tab.num_slack_vars, 4);
        assert_eq!(tab.num_artificial_vars, 0);

        assert_eq!(tab.cur_bounds, vec![10.0, 5.0, 5.0, 10.0]);
        assert_eq!(tab.non_basic_vars, vec![2, 3]);
        assert_eq!(tab.cur_obj, vec![3.0, 4.0]);
        assert_eq!(tab.cur_obj_val, -50.0);

        assert_eq!(tab.basic_vars, vec![0, 1, 4, 5]);
    }

    #[test]
    fn optimize() {
        let mut tab = Tableau::new(
            vec![-3.0, -4.0],
            vec![
                Constraint::Ge(to_sparse(&[1.0, 0.0]), 10.0),
                Constraint::Ge(to_sparse(&[0.0, 1.0]), 5.0),
                Constraint::Le(to_sparse(&[1.0, 1.0]), 20.0),
                Constraint::Le(to_sparse(&[-1.0, 4.0]), 20.0),
            ],
        );

        tab.optimize().unwrap();
        assert_eq!(tab.cur_solution(), vec![12.0, 8.0]);
        assert_eq!(tab.cur_obj_val(), -68.0);
    }

    #[test]
    fn move_to_solution() {
        let mut tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                Constraint::Le(to_sparse(&[1.0, 1.0]), 4.0),
                Constraint::Ge(to_sparse(&[1.0, 1.0]), 2.0),
                Constraint::Eq(to_sparse(&[0.0, 1.0]), 3.0),
            ],
        );

        assert_eq!(
            tab.move_to_solution(&[0.0, 4.0]).err(),
            Some(Error::Infeasible)
        );
        assert_eq!(
            tab.move_to_solution(&[0.5, 3.0]).err(),
            Some(Error::Infeasible)
        );

        assert!(tab.move_to_solution(&[1.0, 3.0]).is_ok());
        assert_eq!(tab.cur_obj_val(), 5.0);

        tab.optimize().unwrap();
        assert_eq!(tab.cur_solution(), vec![0.0, 3.0]);
        assert_eq!(tab.cur_obj_val(), 3.0);
    }

    #[test]
    fn set_unset_var() {
        let mut orig_tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                Constraint::Le(to_sparse(&[1.0, 1.0]), 4.0),
                Constraint::Ge(to_sparse(&[1.0, 1.0]), 2.0),
            ],
        );
        orig_tab.canonicalize().unwrap();
        orig_tab.optimize().unwrap();

        {
            let mut tab = orig_tab.clone();
            tab.set_var(0, 3.0).unwrap();
            assert_eq!(tab.cur_solution(), vec![3.0, 0.0]);
            assert_eq!(tab.cur_obj_val(), 6.0);

            tab.unset_var(0).unwrap();
            assert_eq!(tab.cur_solution(), vec![0.0, 2.0]);
            assert_eq!(tab.cur_obj_val(), 2.0);
        }

        {
            let mut tab = orig_tab.clone();
            tab.set_var(1, 3.0).unwrap();
            assert_eq!(tab.cur_solution(), vec![0.0, 3.0]);
            assert_eq!(tab.cur_obj_val(), 3.0);

            tab.unset_var(1).unwrap();
            assert_eq!(tab.cur_solution(), vec![0.0, 2.0]);
            assert_eq!(tab.cur_obj_val(), 2.0);
        }
    }

    #[test]
    fn add_constraint() {
        let mut orig_tab = Tableau::new(
            vec![2.0, 1.0],
            vec![
                Constraint::Le(to_sparse(&[1.0, 1.0]), 4.0),
                Constraint::Ge(to_sparse(&[1.0, 1.0]), 2.0),
            ],
        );
        orig_tab.canonicalize().unwrap();
        orig_tab.optimize().unwrap();

        {
            let mut tab = orig_tab.clone();
            tab.add_constraints(&[Constraint::Le(to_sparse(&[-1.0, 1.0]), 0.0)])
                .unwrap();
            assert_eq!(tab.cur_solution(), [1.0, 1.0]);
            assert_eq!(tab.cur_obj_val(), 3.0);
        }

        {
            let mut tab = orig_tab.clone();
            tab.set_var(1, 1.5).unwrap();
            tab.add_constraints(&[Constraint::Le(to_sparse(&[-1.0, 1.0]), 0.0)])
                .unwrap();
            assert_eq!(tab.cur_solution(), [1.5, 1.5]);
            assert_eq!(tab.cur_obj_val(), 4.5);
        }
    }
}
