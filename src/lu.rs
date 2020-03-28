use crate::sparse::{Error, Perm, ScatteredVec, SparseMat, TriangleMat};

#[derive(Clone)]
pub struct LUFactors {
    lower: TriangleMat,
    upper: TriangleMat,
    row_perm: Option<Perm>,
    col_perm: Option<Perm>,
}

#[derive(Clone, Debug)]
pub struct ScratchSpace {
    rhs: ScatteredVec,
    dense_rhs: Vec<f64>,
    mark_nonzero: MarkNonzero,
}

impl ScratchSpace {
    pub fn with_capacity(n: usize) -> ScratchSpace {
        ScratchSpace {
            rhs: ScatteredVec::empty(n),
            dense_rhs: vec![0.0; n],
            mark_nonzero: MarkNonzero::with_capacity(n),
        }
    }

    pub(crate) fn clear_sparse(&mut self, size: usize) {
        self.rhs.clear_and_resize(size);
        self.mark_nonzero.clear_and_resize(size);
    }
}

impl std::fmt::Debug for LUFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L:\n{:?}", self.lower)?;
        writeln!(f, "U:\n{:?}", self.upper)?;
        writeln!(
            f,
            "row_perm.new2orig: {:?}",
            self.row_perm.as_ref().map(|p| &p.new2orig)
        )?;
        writeln!(
            f,
            "col_perm.new2orig: {:?}",
            self.col_perm.as_ref().map(|p| &p.new2orig)
        )?;
        Ok(())
    }
}

impl LUFactors {
    pub fn nnz(&self) -> usize {
        self.lower.nondiag.nnz() + self.upper.nondiag.nnz() + self.lower.cols()
    }

    pub fn solve_dense(&self, rhs: &mut [f64], scratch: &mut ScratchSpace) {
        scratch.dense_rhs.resize(rhs.len(), 0.0);

        if let Some(row_perm) = &self.row_perm {
            for i in 0..rhs.len() {
                scratch.dense_rhs[row_perm.orig2new[i]] = rhs[i];
            }
        } else {
            scratch.dense_rhs.copy_from_slice(rhs);
        }

        tri_solve_dense(&self.lower, Triangle::Lower, &mut scratch.dense_rhs);
        tri_solve_dense(&self.upper, Triangle::Upper, &mut scratch.dense_rhs);

        if let Some(col_perm) = &self.col_perm {
            for i in 0..rhs.len() {
                rhs[col_perm.new2orig[i]] = scratch.dense_rhs[i];
            }
        } else {
            rhs.copy_from_slice(&mut scratch.dense_rhs);
        }
    }

    pub fn solve(&self, rhs: &mut ScatteredVec, scratch: &mut ScratchSpace) {
        if let Some(row_perm) = &self.row_perm {
            scratch.rhs.clear();
            for &i in &rhs.nonzero {
                let new_i = row_perm.orig2new[i];
                scratch.rhs.nonzero.push(new_i);
                scratch.rhs.is_nonzero[new_i] = true;
                scratch.rhs.values[new_i] = rhs.values[i];
            }
        } else {
            std::mem::swap(&mut scratch.rhs, rhs);
        }

        tri_solve_sparse(&self.lower, scratch);
        tri_solve_sparse(&self.upper, scratch);

        if let Some(col_perm) = &self.col_perm {
            rhs.clear();
            for &i in &scratch.rhs.nonzero {
                let new_i = col_perm.new2orig[i];
                rhs.nonzero.push(new_i);
                rhs.is_nonzero[new_i] = true;
                rhs.values[new_i] = scratch.rhs.values[i];
            }
        } else {
            std::mem::swap(rhs, &mut scratch.rhs);
        }
    }

    pub fn transpose(&self) -> LUFactors {
        LUFactors {
            lower: self.upper.transpose(),
            upper: self.lower.transpose(),
            row_perm: self.col_perm.clone(),
            col_perm: self.row_perm.clone(),
        }
    }
}

pub fn lu_factorize<'a>(
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
) -> Result<LUFactors, Error> {
    // Implementation of the Gilbert-Peierls algorithm:
    //
    // Gilbert, John R., and Tim Peierls. "Sparse partial pivoting in time
    // proportional to arithmetic operations." SIAM Journal on Scientific and
    // Statistical Computing 9.5 (1988): 862-874.
    //
    // https://ecommons.cornell.edu/bitstream/handle/1813/6623/86-783.pdf

    let mat_nnz = (0..size).map(|c| get_col(c).0.len()).sum::<usize>();
    trace!(
        "lu_factorize: starting, matrix size: {}, nnz: {} (excess: {})",
        size,
        mat_nnz,
        mat_nnz - size,
    );

    let col_perm = super::ordering::order_simple(size, |c| get_col(c).0);

    let mut orig_row2elt_count = vec![0; size];
    for col_rows in (0..size).map(|c| get_col(c).0) {
        for &orig_r in col_rows {
            orig_row2elt_count[orig_r] += 1;
        }
    }

    scratch.clear_sparse(size);

    let mut lower = SparseMat::new(size);
    let mut upper = SparseMat::new(size);
    let mut upper_diag = Vec::with_capacity(size);

    let mut new2orig_row = (0..size).collect::<Vec<_>>();
    let mut orig2new_row = new2orig_row.clone();

    for i_col in 0..size {
        let mat_col = get_col(col_perm.new2orig[i_col]);

        // Solve the equation L'_j * x = a_j (x will be in scratch.rhs).
        // L'_j is a sq. matrix with the first j columns of L
        // and columns of identity matrix after j.
        // Part of x above the diagonal (in the new row indices) is the column of U
        // and part of x below the diagonal divided by the pivot value is the column of L

        scratch.rhs.set(mat_col.0.iter().copied().zip(mat_col.1));

        scratch.mark_nonzero.run(
            &mut scratch.rhs,
            |new_i| &lower.col_rows(new_i),
            |new_i| new_i < i_col,
            |orig_r| orig2new_row[orig_r],
        );

        // At this point all future nonzero positions of scratch.rhs are marked
        // and the order in which variables depend on each other is determined.
        // rev() because DFS returns vertices in reverse topological order.
        for &orig_i in scratch.mark_nonzero.visited.iter().rev() {
            // values[orig_i] is already fully calculated, diag coeff = 1.0.
            let new_i = orig2new_row[orig_i];
            if new_i < i_col {
                let x_val = scratch.rhs.values[orig_i];
                for (orig_r, coeff) in lower.col_iter(new_i) {
                    scratch.rhs.values[orig_r] -= x_val * coeff;
                }
            }
        }

        // Next we choose a pivot among values of x below the diagonal.
        // Pivoting by choosing the max element is good for stability,
        // but bad for sparseness, so we do threshold pivoting instead.

        let pivot_orig_r = {
            let mut max_abs = 0.0;
            for &orig_r in &scratch.rhs.nonzero {
                if orig2new_row[orig_r] < i_col {
                    continue;
                }

                let abs = f64::abs(scratch.rhs.values[orig_r]);
                if abs > max_abs {
                    max_abs = abs;
                }
            }

            if max_abs < 1e-8 {
                return Err(Error::SingularMatrix);
            }

            assert!(max_abs.is_normal());

            // Choose among eligible pivot rows one with the least elements.
            // Gilbert-Peierls suggest to choose row with least elements *to the right*,
            // but it yielded poor results. Our heuristic is not a huge improvement either,
            // but at least we are less dependent on initial row ordering.
            let mut best_orig_r = None;
            let mut best_elt_count = None;
            for &orig_r in &scratch.rhs.nonzero {
                if orig2new_row[orig_r] < i_col {
                    continue;
                }

                if f64::abs(scratch.rhs.values[orig_r]) >= stability_coeff * max_abs {
                    let elt_count = orig_row2elt_count[orig_r];
                    if best_elt_count.is_none() || best_elt_count.unwrap() > elt_count {
                        best_orig_r = Some(orig_r);
                        best_elt_count = Some(elt_count);
                    }
                }
            }
            best_orig_r.unwrap()
        };

        let pivot_val = scratch.rhs.values[pivot_orig_r];

        {
            // Keep track of row permutations.
            let row = i_col;
            let orig_row = new2orig_row[row];
            let pivot_row = orig2new_row[pivot_orig_r];
            new2orig_row.swap(row, pivot_row);
            orig2new_row.swap(orig_row, pivot_orig_r);
        }

        // Gather the values of x into lower and upper matrices.

        for &orig_r in &scratch.rhs.nonzero {
            let val = scratch.rhs.values[orig_r];

            if val == 0.0 {
                continue;
            }

            let new_r = orig2new_row[orig_r];
            if new_r < i_col {
                upper.push(new_r, val);
            } else if new_r == i_col {
                upper_diag.push(pivot_val);
            } else {
                lower.push(orig_r, val / pivot_val);
            }
        }

        upper.seal_column();
        lower.seal_column();
    }

    // permute rows of lower to "new" indices.
    for i_col in 0..lower.cols() {
        for r in lower.col_rows_mut(i_col) {
            *r = orig2new_row[*r];
        }
    }

    let lower_nnz = lower.nnz();
    let upper_nnz = upper.nnz();
    trace!(
        "lu_factorize: done, lower nnz: {} (excess: {}), upper nnz: {} (excess: {}), additional fill-in: {}",
        lower_nnz + size,
        lower_nnz,
        upper_nnz + size,
        upper_nnz,
        lower_nnz + upper_nnz + size - mat_nnz,
    );

    let res = LUFactors {
        lower: TriangleMat {
            nondiag: lower,
            diag: None,
        },
        upper: TriangleMat {
            nondiag: upper,
            diag: Some(upper_diag),
        },
        row_perm: Some(Perm {
            orig2new: orig2new_row,
            new2orig: new2orig_row,
        }),
        col_perm: Some(col_perm),
    };

    Ok(res)
}

#[derive(Clone, Debug)]
struct MarkNonzero {
    dfs_stack: Vec<DfsStep>,
    is_visited: Vec<bool>,
    visited: Vec<usize>, // in reverse topological order
}

#[derive(Clone, Debug)]
struct DfsStep {
    orig_i: usize,
    cur_child: usize,
}

impl MarkNonzero {
    fn with_capacity(n: usize) -> MarkNonzero {
        MarkNonzero {
            dfs_stack: Vec::with_capacity(n),
            is_visited: vec![false; n],
            visited: vec![],
        }
    }

    fn clear(&mut self) {
        assert!(self.dfs_stack.is_empty());
        for &i in &self.visited {
            self.is_visited[i] = false;
        }
        self.visited.clear();
    }

    fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.dfs_stack.reserve(n);
        self.is_visited.resize(n, false);
    }

    // compute the non-zero elements of the result by dfs traversal
    fn run<'a>(
        &mut self,
        rhs: &mut ScatteredVec,
        get_children: impl Fn(usize) -> &'a [usize] + 'a,
        filter: impl Fn(usize) -> bool,
        orig2new_row: impl Fn(usize) -> usize,
    ) {
        self.clear();

        for &orig_r in &rhs.nonzero {
            let new_r = orig2new_row(orig_r);
            if !filter(new_r) {
                continue;
            }
            if self.is_visited[orig_r] {
                continue;
            }

            self.dfs_stack.push(DfsStep {
                orig_i: orig_r,
                cur_child: 0,
            });
            while !self.dfs_stack.is_empty() {
                let cur_step = self.dfs_stack.last_mut().unwrap();
                let new_i = orig2new_row(cur_step.orig_i);
                let children = if filter(new_i) {
                    get_children(new_i)
                } else {
                    &[]
                };
                if !self.is_visited[cur_step.orig_i] {
                    self.is_visited[cur_step.orig_i] = true;
                } else {
                    cur_step.cur_child += 1;
                }

                while cur_step.cur_child < children.len() {
                    let child_orig_r = children[cur_step.cur_child];
                    if !self.is_visited[child_orig_r] {
                        break;
                    }
                    cur_step.cur_child += 1;
                }

                if cur_step.cur_child < children.len() {
                    let i_child = cur_step.cur_child;
                    self.dfs_stack.push(DfsStep {
                        orig_i: children[i_child],
                        cur_child: 0,
                    });
                } else {
                    self.visited.push(cur_step.orig_i);
                    self.dfs_stack.pop();
                }
            }
        }

        for &i in &self.visited {
            if !rhs.is_nonzero[i] {
                rhs.is_nonzero[i] = true;
                rhs.nonzero.push(i)
            }
        }
    }
}

enum Triangle {
    Lower,
    Upper,
}

fn tri_solve_dense(tri_mat: &TriangleMat, triangle: Triangle, rhs: &mut [f64]) {
    assert_eq!(tri_mat.rows(), rhs.len());
    match triangle {
        Triangle::Lower => {
            for col in 0..tri_mat.cols() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }

        Triangle::Upper => {
            for col in (0..tri_mat.cols()).rev() {
                tri_solve_process_col(tri_mat, col, rhs);
            }
        }
    };
}

/// rhs is passed via scratch.visited, scratch.values.
fn tri_solve_sparse(tri_mat: &TriangleMat, scratch: &mut ScratchSpace) {
    assert_eq!(tri_mat.rows(), scratch.rhs.len());

    // compute the non-zero elements of the result by dfs traversal
    scratch.mark_nonzero.run(
        &mut scratch.rhs,
        |col| tri_mat.nondiag.col_rows(col),
        |_| true,
        |orig_i| orig_i,
    );

    // solve for the non-zero values into dense workspace.
    // rev() because DFS returns vertices in reverse topological order.
    for &col in scratch.mark_nonzero.visited.iter().rev() {
        tri_solve_process_col(tri_mat, col, &mut scratch.rhs.values);
    }
}

fn tri_solve_process_col(tri_mat: &TriangleMat, col: usize, rhs: &mut [f64]) {
    // all other variables in this row (multiplied by their coeffs)
    // are already subtracted from rhs[col].
    let x_val = if let Some(diag) = tri_mat.diag.as_ref() {
        rhs[col] / diag[col]
    } else {
        rhs[col]
    };

    rhs[col] = x_val;
    for (r, &coeff) in tri_mat.nondiag.col_iter(col) {
        rhs[r] -= x_val * coeff;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{assert_matrix_eq, to_dense, to_sparse};
    use sprs::{CsMat, CsVec, TriMat};

    fn mat_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> CsMat<f64> {
        let mut mat = TriMat::with_capacity((rows, cols), triplets.len());
        for (r, c, val) in triplets {
            mat.add_triplet(*r, *c, *val);
        }
        mat.to_csc()
    }

    #[test]
    fn lu_simple() {
        let mat = mat_from_triplets(
            3,
            4,
            &[
                (0, 1, 2.0),
                (0, 0, 2.0),
                (0, 2, 123.0),
                (1, 2, 456.0),
                (1, 3, 1.0),
                (2, 1, 4.0),
                (2, 0, 3.0),
                (2, 2, 789.0),
                (2, 3, 1.0),
            ],
        );

        let mut scratch = ScratchSpace::with_capacity(mat.rows());
        let lu = lu_factorize(
            mat.rows(),
            |c| mat.outer_view([1, 0, 3][c]).unwrap().into_raw_storage(),
            0.9,
            &mut scratch,
        )
        .unwrap();
        let lu_transp = lu.transpose();

        let l_nondiag_ref = [
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ];
        assert_matrix_eq(&lu.lower.nondiag.to_csmat(), &l_nondiag_ref);
        assert_eq!(lu.lower.diag, None);

        let u_nondiag_ref = [
            vec![0.0, 3.0, 1.0],
            vec![0.0, 0.0, -0.5],
            vec![0.0, 0.0, 0.0],
        ];
        let u_diag_ref = [4.0, 0.5, 1.0];
        assert_matrix_eq(&lu.upper.nondiag.to_csmat(), &u_nondiag_ref);
        assert_eq!(lu.upper.diag.as_ref().unwrap(), &u_diag_ref);

        assert_eq!(lu.row_perm.as_ref().unwrap().new2orig, &[2, 0, 1]);
        assert_eq!(lu.col_perm.as_ref().unwrap().new2orig, &[0, 1, 2]);

        {
            let mut rhs_dense = [6.0, 3.0, 13.0];
            lu.solve_dense(&mut rhs_dense, &mut scratch);
            assert_eq!(&rhs_dense, &[1.0, 2.0, 3.0]);
        }

        {
            let mut rhs_dense_t = [14.0, 11.0, 5.0];
            lu_transp.solve_dense(&mut rhs_dense_t, &mut scratch);
            assert_eq!(&rhs_dense_t, &[1.0, 2.0, 3.0]);
        }

        {
            let mut rhs = ScatteredVec::empty(3);
            rhs.set(to_sparse(&[0.0, -1.0, 0.0]).iter());
            lu.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![1.0, -1.0, -1.0]);
        }

        {
            let mut rhs = ScatteredVec::empty(3);
            rhs.set(to_sparse(&[0.0, -1.0, 1.0]).iter());
            lu_transp.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![-2.0, 0.0, 1.0]);
        }
    }

    #[test]
    fn lu_singular() {
        let size = 3;

        {
            let symbolically_singular = mat_from_triplets(
                size,
                size,
                &[(0, 0, 1.0), (1, 0, 1.0), (1, 1, 2.0), (1, 2, 3.0)],
            );

            let mut scratch = ScratchSpace::with_capacity(size);
            let err = lu_factorize(
                size,
                |c| {
                    symbolically_singular
                        .outer_view(c)
                        .unwrap()
                        .into_raw_storage()
                },
                0.9,
                &mut scratch,
            );
            assert_eq!(err.unwrap_err(), Error::SingularMatrix);
        }

        {
            let numerically_singular = mat_from_triplets(
                size,
                size,
                &[
                    (0, 0, 1.0),
                    (1, 0, 1.0),
                    (1, 1, 2.0),
                    (1, 2, 3.0),
                    (2, 0, 2.0),
                    (2, 1, 2.0),
                    (2, 2, 3.0),
                ],
            );

            let mut scratch = ScratchSpace::with_capacity(size);
            let err = lu_factorize(
                size,
                |c| {
                    numerically_singular
                        .outer_view(c)
                        .unwrap()
                        .into_raw_storage()
                },
                0.9,
                &mut scratch,
            );
            assert_eq!(err.unwrap_err(), Error::SingularMatrix);
        }
    }

    #[test]
    fn lu_rand() {
        let size = 10;

        let mut rng = rand_pcg::Pcg64::seed_from_u64(12345);
        use rand::prelude::*;

        let mut mat = TriMat::new((size, size));
        for r in 0..size {
            for c in 0..size {
                if rng.gen_range(0, 2) == 0 {
                    mat.add_triplet(r, c, rng.gen_range(0.0, 1.0));
                }
            }
        }
        let mat = mat.to_csc();

        let mut scratch = ScratchSpace::with_capacity(mat.rows());

        // TODO: random permutation?
        let cols: Vec<_> = (0..size).collect();

        let lu = lu_factorize(
            size,
            |c| mat.outer_view(cols[c]).unwrap().into_raw_storage(),
            0.1,
            &mut scratch,
        )
        .unwrap();
        let lu_transp = lu.transpose();

        let multiplied = &lu.lower.to_csmat() * &lu.upper.to_csmat();
        assert!(multiplied.is_csc());
        for (i, &c) in cols.iter().enumerate() {
            let permuted = {
                let mut is = vec![];
                let mut vs = vec![];
                for (i, &val) in mat.outer_view(c).unwrap().iter() {
                    is.push(lu.row_perm.as_ref().unwrap().orig2new[i]);
                    vs.push(val);
                }
                CsVec::new(size, is, vs)
            };
            let diff = &multiplied
                .outer_view(lu.col_perm.as_ref().unwrap().orig2new[i])
                .unwrap()
                - &permuted;
            assert!(diff.norm(1.0) < 1e-5);
        }

        type ArrayVec = ndarray::Array1<f64>;
        let dense_rhs: Vec<_> = (0..size).map(|_| rng.gen_range(0.0, 1.0)).collect();

        {
            let mut dense_sol = dense_rhs.clone();
            lu.solve_dense(&mut dense_sol, &mut scratch);
            let diff = &ArrayVec::from(dense_rhs.clone()) - &(&mat * &ArrayVec::from(dense_sol));
            assert!(f64::sqrt(diff.dot(&diff)) < 1e-5);
        }

        {
            let mut dense_sol_t = dense_rhs.clone();
            lu_transp.solve_dense(&mut dense_sol_t, &mut scratch);
            let diff = &ArrayVec::from(dense_rhs)
                - &(&mat.transpose_view() * &ArrayVec::from(dense_sol_t));
            assert!(f64::sqrt(diff.dot(&diff)) < 1e-5);
        }

        let sparse_rhs = {
            let mut res = CsVec::empty(size);
            for i in 0..size {
                if rng.gen_range(0, 3) == 0 {
                    res.append(i, rng.gen_range(0.0, 1.0));
                }
            }
            res
        };

        {
            let mut rhs = ScatteredVec::empty(size);
            rhs.set(sparse_rhs.iter());
            lu.solve(&mut rhs, &mut scratch);
            let diff = &sparse_rhs - &(&mat * &rhs.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }

        {
            let mut rhs_t = ScatteredVec::empty(size);
            rhs_t.set(sparse_rhs.iter());
            lu_transp.solve(&mut rhs_t, &mut scratch);
            let diff = &sparse_rhs - &(&mat.transpose_view() * &rhs_t.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }
    }
}
