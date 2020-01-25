use crate::helpers::to_dense;
#[cfg(test)]
use sprs::CsVec;
use sprs::{CsMat, CsVecView};

#[derive(Clone)]
pub struct LUFactors {
    lower: CscMat,
    upper: CscMat,
    row_perm: Option<Perm>,
    col_perm: Option<Perm>,
}

/// Unordered sparse matrix with elements stored by columns
#[derive(Clone, Debug)]
pub struct CscMat {
    n_rows: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl CscMat {
    pub fn new(n_rows: usize) -> CscMat {
        CscMat {
            n_rows,
            indptr: vec![0],
            indices: vec![],
            data: vec![],
        }
    }

    pub fn rows(&self) -> usize {
        self.n_rows
    }

    pub fn cols(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn clear_and_resize(&mut self, n_rows: usize) {
        self.data.clear();
        self.indices.clear();
        self.indptr.clear();
        self.indptr.push(0);
        self.n_rows = n_rows;
    }

    fn push(&mut self, row: usize, val: f64) {
        self.indices.push(row);
        self.data.push(val);
    }

    fn seal_column(&mut self) {
        self.indptr.push(self.indices.len())
    }

    pub fn col_rows(&self, i_col: usize) -> &[usize] {
        &self.indices[self.indptr[i_col]..self.indptr[i_col + 1]]
    }

    pub fn col_rows_mut(&mut self, i_col: usize) -> &mut [usize] {
        &mut self.indices[self.indptr[i_col]..self.indptr[i_col + 1]]
    }

    pub fn col_data(&self, i_col: usize) -> &[f64] {
        &self.data[self.indptr[i_col]..self.indptr[i_col + 1]]
    }

    pub fn col_iter(&self, i_col: usize) -> impl Iterator<Item = (usize, &f64)> {
        self.col_rows(i_col)
            .iter()
            .copied()
            .zip(self.col_data(i_col))
    }

    pub fn append_scattered_col(&mut self, col: &ScatteredVec) {
        self.indices.extend_from_slice(&col.nonzero);
        self.data.extend(col.iter().map(|(_, &val)| val));
        self.seal_column();
    }

    pub fn into_csmat(self) -> CsMat<f64> {
        CsMat::new_csc(
            (self.cols(), self.n_rows),
            self.indptr,
            self.indices,
            self.data,
        )
    }

    pub fn to_csmat(&self) -> CsMat<f64> {
        self.clone().into_csmat()
    }

    pub fn transpose(&self) -> CscMat {
        let mut out = CscMat {
            n_rows: self.cols(),
            indptr: vec![],
            indices: vec![],
            data: vec![],
        };

        // calculate row counts and store them in the indptr array.
        out.indptr.clear();
        out.indptr.resize(self.rows() + 1, 0);
        for c in 0..self.cols() {
            for &r in self.col_rows(c) {
                out.indptr[r] += 1;
            }
        }

        // calculate cumulative counts so that indptr elements point to
        // the *ends* of each resulting row.
        for r in 1..out.indptr.len() {
            out.indptr[r] += out.indptr[r - 1];
        }

        // place the elements
        out.indices.clear();
        out.indices.resize(self.nnz(), 0);
        out.data.clear();
        out.data.resize(self.nnz(), 0.0);
        for c in 0..self.cols() {
            for (r, &val) in self.col_iter(c) {
                out.indptr[r] -= 1;
                out.indices[out.indptr[r]] = c;
                out.data[out.indptr[r]] = val;
            }
        }

        *out.indptr.last_mut().unwrap() = self.nnz();

        out
    }
}

#[derive(Clone)]
pub struct Perm {
    orig2new: Vec<usize>,
    new2orig: Vec<usize>,
}

impl Perm {
    fn inv(self) -> Perm {
        Perm {
            orig2new: self.new2orig,
            new2orig: self.orig2new,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScatteredVec {
    values: Vec<f64>,
    is_nonzero: Vec<bool>,
    nonzero: Vec<usize>,
}

impl ScatteredVec {
    pub fn empty(n: usize) -> ScatteredVec {
        ScatteredVec {
            values: vec![0.0; n],
            is_nonzero: vec![false; n],
            nonzero: vec![],
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (usize, &'a f64)> {
        self.nonzero.iter().map(move |&i| (i, &self.values[i]))
    }

    pub fn clear(&mut self) {
        for &i in &self.nonzero {
            self.values[i] = 0.0;
            self.is_nonzero[i] = false;
        }
        self.nonzero.clear();
    }

    pub fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.values.resize(n, 0.0);
        self.is_nonzero.resize(n, false);
    }

    pub fn set(&mut self, rhs: CsVecView<f64>) {
        self.clear();
        for (i, &val) in rhs.iter() {
            self.is_nonzero[i] = true;
            self.nonzero.push(i);
            self.values[i] = val;
        }
    }

    #[cfg(test)]
    pub fn to_csvec(&self) -> CsVec<f64> {
        let mut indices = vec![];
        let mut data = vec![];
        for &i in &self.nonzero {
            let val = self.values[i];
            if val != 0.0 {
                indices.push(i);
                data.push(val);
            }
        }
        CsVec::new(self.values.len(), indices, data)
    }

    #[inline]
    pub fn get(&mut self, i: usize) -> &f64 {
        &self.values[i]
    }

    #[inline]
    pub fn get_mut(&mut self, i: usize) -> &mut f64 {
        if !self.is_nonzero[i] {
            self.is_nonzero[i] = true;
            self.nonzero.push(i);
        }
        &mut self.values[i]
    }
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
}

impl std::fmt::Debug for LUFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "L:\n")?;
        for row in self.lower.to_csmat().to_csr().outer_iterator() {
            write!(f, "{:?}\n", to_dense(&row))?
        }
        write!(f, "U:\n")?;
        for row in self.upper.to_csmat().to_csr().outer_iterator() {
            write!(f, "{:?}\n", to_dense(&row))?
        }
        write!(
            f,
            "row_perm.new2orig: {:?}\n",
            self.row_perm.as_ref().map(|p| &p.new2orig)
        )?;
        write!(
            f,
            "col_perm.new2orig: {:?}\n",
            self.col_perm.as_ref().map(|p| &p.new2orig)
        )?;

        Ok(())
    }
}

impl LUFactors {
    pub fn nnz(&self) -> usize {
        self.lower.nnz() + self.upper.nnz()
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
                rhs[col_perm.orig2new[i]] = scratch.dense_rhs[i];
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
                let new_i = col_perm.orig2new[i];
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
            row_perm: self.col_perm.as_ref().map(|p| p.clone().inv()),
            col_perm: self.row_perm.as_ref().map(|p| p.clone().inv()),
        }
    }
}

pub fn lu_factorize(
    mat: &CsMat<f64>,
    cols: &[usize],
    stability_coeff: f64,
    scratch: &mut ScratchSpace,
) -> LUFactors {
    assert!(mat.is_csc());
    assert_eq!(mat.rows(), cols.len());

    trace!(
        "lu_factorize: starting, matrix nnz: {}",
        cols.iter()
            .map(|&c| mat.outer_view(c).unwrap().nnz())
            .sum::<usize>()
    );

    let mut orig_row2elt_count = vec![0; mat.rows()];
    for &c in cols {
        let col = mat.outer_view(c).unwrap();
        for (orig_r, _) in col.iter() {
            orig_row2elt_count[orig_r] += 1;
        }
    }

    scratch.rhs.clear_and_resize(cols.len());
    scratch.mark_nonzero.clear_and_resize(cols.len());

    let mut lower = CscMat::new(mat.rows());
    let mut upper = CscMat::new(mat.rows());

    let mut new2orig_row = (0..mat.rows()).collect::<Vec<_>>();
    let mut orig2new_row = new2orig_row.clone();

    for i_col in 0..cols.len() {
        scratch.rhs.clear();
        let mat_col = mat.outer_view(cols[i_col]).unwrap();

        // 3. calculate i-th column of U
        for (orig_r, &val) in mat_col.iter() {
            if orig2new_row[orig_r] < i_col {
                scratch.rhs.values[orig_r] = val;
                scratch.rhs.is_nonzero[orig_r] = true;
                scratch.rhs.nonzero.push(orig_r);
            }
        }

        scratch.mark_nonzero.run(
            &mut scratch.rhs,
            |new_i| &lower.col_rows(new_i),
            |new_i| new_i < i_col,
            |orig_r| orig2new_row[orig_r],
        );

        // rev() because DFS returns vertices in reverse topological order.
        for &orig_i in scratch.mark_nonzero.visited.iter().rev() {
            // values[orig_i] is already fully calculated, diag coeff = 1.0.
            let x_val = scratch.rhs.values[orig_i];
            let new_i = orig2new_row[orig_i];
            for (orig_r, coeff) in lower.col_iter(new_i) {
                let new_r = orig2new_row[orig_r];
                if new_r < i_col && new_r > new_i {
                    scratch.rhs.values[orig_r] -= x_val * coeff;
                }
            }
        }

        // 4.
        // Now calculate b vector in scratch.rhs.
        // It will occupy different indices than the new U col.
        let below_rows_start = scratch.rhs.nonzero.len();
        for (orig_r, &val) in mat_col.iter() {
            if orig2new_row[orig_r] >= i_col {
                scratch.rhs.values[orig_r] = val;
                scratch.rhs.is_nonzero[orig_r] = true;
                scratch.rhs.nonzero.push(orig_r);
            }
        }

        for i_upper in 0..below_rows_start {
            let orig_u_r = scratch.rhs.nonzero[i_upper];
            let u_coeff = scratch.rhs.values[orig_u_r];

            if u_coeff != 0.0 {
                let new_u_r = orig2new_row[orig_u_r];
                upper.push(new_u_r, u_coeff);

                for (orig_r, val) in lower.col_iter(new_u_r) {
                    if orig2new_row[orig_r] >= i_col {
                        if !scratch.rhs.is_nonzero[orig_r] {
                            scratch.rhs.is_nonzero[orig_r] = true;
                            scratch.rhs.nonzero.push(orig_r);
                        }
                        scratch.rhs.values[orig_r] -= val * u_coeff;
                    }
                }
            }
        }

        // Index of the pivot element in tmp_below_rows array.
        // Pivoting by choosing the max element is good for stability,
        // but bad for sparseness, so we do threshold pivoting instead.

        let pivot_i = {
            let mut max_abs = 0.0;
            for &orig_r in &scratch.rhs.nonzero[below_rows_start..] {
                let abs = f64::abs(scratch.rhs.values[orig_r]);
                if abs > max_abs {
                    max_abs = abs;
                }
            }
            assert!(max_abs.is_normal());

            // Choose among eligible pivot rows one with the least elements.
            // Gilbert-Peierls suggest to choose row with least elements *to the right*,
            // but it yielded poor results. Our heuristic is not a huge improvement either,
            // but at least we are less dependent on initial row ordering.
            let mut best_i = None;
            let mut best_elt_count = None;
            for i in below_rows_start..scratch.rhs.nonzero.len() {
                let orig_r = scratch.rhs.nonzero[i];
                if f64::abs(scratch.rhs.values[orig_r]) >= stability_coeff * max_abs {
                    let elt_count = orig_row2elt_count[orig_r];
                    if best_elt_count.is_none() || best_elt_count.unwrap() > elt_count {
                        best_i = Some(i);
                        best_elt_count = Some(elt_count);
                    }
                }
            }
            best_i.unwrap()
        };

        let pivot_val = scratch.rhs.values[scratch.rhs.nonzero[pivot_i]];

        // 5.
        {
            let row = i_col;
            let orig_row = new2orig_row[row];
            let orig_pivot_row = scratch.rhs.nonzero[pivot_i];
            let pivot_row = orig2new_row[orig_pivot_row];
            new2orig_row.swap(row, pivot_row);
            orig2new_row.swap(orig_row, orig_pivot_row);
        }

        // 6, 7.

        // diagonal elements.
        upper.push(i_col, pivot_val);
        upper.seal_column();
        lower.push(scratch.rhs.nonzero[pivot_i], 1.0);

        for i in below_rows_start..scratch.rhs.nonzero.len() {
            let orig_row = scratch.rhs.nonzero[i];
            let val = scratch.rhs.values[orig_row];

            if val == 0.0 {
                continue;
            }

            if i != pivot_i {
                lower.push(orig_row, val / pivot_val);
            }
        }
        lower.seal_column();
    }

    // permute rows of lower to "new" indices.
    for i_col in 0..lower.cols() {
        for r in lower.col_rows_mut(i_col) {
            *r = orig2new_row[*r];
        }
    }

    trace!(
        "lu_factorize: done, lower nnz: {}, upper nnz: {}",
        lower.nnz(),
        upper.nnz()
    );

    LUFactors {
        lower,
        upper,
        row_perm: Some(Perm {
            orig2new: orig2new_row,
            new2orig: new2orig_row,
        }),
        col_perm: None,
    }
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
                let children = get_children(orig2new_row(cur_step.orig_i));
                if !self.is_visited[cur_step.orig_i] {
                    self.is_visited[cur_step.orig_i] = true;
                } else {
                    cur_step.cur_child += 1;
                }

                while cur_step.cur_child < children.len() {
                    let child_orig_r = children[cur_step.cur_child];
                    let child_new_r = orig2new_row(child_orig_r);
                    if !self.is_visited[child_orig_r] && filter(child_new_r) {
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

fn tri_solve_dense(tri_mat: &CscMat, triangle: Triangle, rhs: &mut [f64]) {
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
fn tri_solve_sparse(tri_mat: &CscMat, scratch: &mut ScratchSpace) {
    assert_eq!(tri_mat.rows(), scratch.rhs.len());

    // compute the non-zero elements of the result by dfs traversal
    scratch.mark_nonzero.run(
        &mut scratch.rhs,
        |col| tri_mat.col_rows(col),
        |_| true,
        |orig_i| orig_i,
    );

    // solve for the non-zero values into dense workspace.
    // rev() because DFS returns vertices in reverse topological order.
    for &col in scratch.mark_nonzero.visited.iter().rev() {
        tri_solve_process_col(tri_mat, col, &mut scratch.rhs.values);
    }
}

fn tri_solve_process_col(tri_mat: &CscMat, col: usize, rhs: &mut [f64]) {
    // TODO: maybe store diag elements separately so that there is no need to seek for them.
    let i_diag = tri_mat
        .col_rows(col)
        .iter()
        .position(|&r| r == col)
        .unwrap();
    let diag_coeff = tri_mat.col_data(col)[i_diag];

    // rhs[col] is already fully calculated.
    let x_val = rhs[col] / diag_coeff;
    for (r, &coeff) in tri_mat.col_iter(col) {
        if r == col {
            rhs[r] = x_val;
        } else {
            rhs[r] -= x_val * coeff;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{assert_matrix_eq, to_sparse};
    use sprs::TriMat;

    #[test]
    fn mat_transpose() {
        let mut mat = CscMat::new(2);
        mat.push(0, 1.1);
        mat.push(1, 2.2);
        mat.seal_column();
        mat.push(1, 3.3);
        mat.seal_column();
        mat.push(0, 4.4);
        mat.seal_column();

        let transp = mat.transpose();
        assert_eq!(&transp.indptr, &[0, 2, 4]);
        assert_eq!(&transp.indices, &[2, 0, 1, 0]);
        assert_eq!(&transp.data, &[4.4, 1.1, 3.3, 2.2]);
    }

    #[test]
    fn lu_simple() {
        let triplets = [
            (0, 1, 2.0),
            (0, 0, 2.0),
            (0, 2, 123.0),
            (1, 2, 456.0),
            (1, 3, 1.0),
            (2, 1, 4.0),
            (2, 0, 3.0),
            (2, 2, 789.0),
            (2, 3, 1.0),
        ];
        let mut mat = TriMat::with_capacity((3, 4), triplets.len());
        for (r, c, val) in &triplets {
            mat.add_triplet(*r, *c, *val);
        }
        let mat = mat.to_csc();
        let mut scratch = ScratchSpace::with_capacity(mat.rows());
        let lu = lu_factorize(&mat, &[1, 0, 3], 0.9, &mut scratch);
        let lu_transp = lu.transpose();

        let l_ref = [
            vec![1.0, 0.0, 0.0],
            vec![0.5, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert_matrix_eq(&lu.lower.to_csmat(), &l_ref);

        let u_ref = [
            vec![4.0, 3.0, 1.0],
            vec![0.0, 0.5, -0.5],
            vec![0.0, 0.0, 1.0],
        ];
        assert_matrix_eq(&lu.upper.to_csmat(), &u_ref);

        assert_eq!(lu.row_perm.as_ref().unwrap().orig2new, &[1, 2, 0]);
        assert_eq!(lu.row_perm.as_ref().unwrap().new2orig, &[2, 0, 1]);

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
            rhs.set(to_sparse(&[0.0, -1.0, 0.0]).view());
            lu.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![1.0, -1.0, -1.0]);
        }

        {
            let mut rhs = ScatteredVec::empty(3);
            rhs.set(to_sparse(&[0.0, -1.0, 1.0]).view());
            lu_transp.solve(&mut rhs, &mut scratch);
            assert_eq!(to_dense(&rhs.to_csvec()), vec![-2.0, 0.0, 1.0]);
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

        let lu = lu_factorize(&mat, &cols, 0.1, &mut scratch);
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
            let diff = &multiplied.outer_view(i).unwrap() - &permuted;
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
            rhs.set(sparse_rhs.view());
            lu.solve(&mut rhs, &mut scratch);
            let diff = &sparse_rhs - &(&mat * &rhs.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }

        {
            let mut rhs_t = ScatteredVec::empty(size);
            rhs_t.set(sparse_rhs.view());
            lu_transp.solve(&mut rhs_t, &mut scratch);
            let diff = &sparse_rhs - &(&mat.transpose_view() * &rhs_t.to_csvec());
            assert!(diff.norm(1.0) < 1e-5);
        }
    }
}
