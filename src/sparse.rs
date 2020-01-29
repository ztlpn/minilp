use sprs::{CsMat, CsVec};

#[derive(Clone, Debug)]
pub struct SparseVec {
    indices: Vec<usize>,
    values: Vec<f64>,
}

impl SparseVec {
    pub fn new() -> SparseVec {
        SparseVec {
            indices: vec![],
            values: vec![],
        }
    }

    pub fn clear(&mut self) {
        self.indices.clear();
        self.values.clear();
    }

    pub fn push(&mut self, i: usize, val: f64) {
        self.indices.push(i);
        self.values.push(val);
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, &f64)> {
        self.indices.iter().copied().zip(&self.values)
    }

    pub fn sq_norm(&self) -> f64 {
        self.values.iter().map(|&v| v * v).sum()
    }

    pub fn into_csvec(self, len: usize) -> CsVec<f64> {
        CsVec::new(len, self.indices, self.values)
    }
}

#[derive(Clone, Debug)]
pub struct ScatteredVec {
    pub(crate) values: Vec<f64>,
    pub(crate) is_nonzero: Vec<bool>,
    pub(crate) nonzero: Vec<usize>,
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

    pub fn iter(&self) -> impl Iterator<Item = (usize, &f64)> {
        self.nonzero.iter().map(move |&i| (i, &self.values[i]))
    }

    pub fn sq_norm(&self) -> f64 {
        self.nonzero
            .iter()
            .map(|&i| self.values[i] * self.values[i])
            .sum()
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

    pub fn set<'a, T>(&mut self, rhs: T)
    where
        T: IntoIterator<Item = (usize, &'a f64)>,
    {
        self.clear();
        for (i, &val) in rhs {
            self.is_nonzero[i] = true;
            self.nonzero.push(i);
            self.values[i] = val;
        }
    }

    pub fn to_sparse_vec(&self, lhs: &mut SparseVec) {
        lhs.clear();
        for &idx in &self.nonzero {
            lhs.indices.push(idx);
            lhs.values.push(self.values[idx])
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

/// Unordered sparse matrix with elements stored by columns
#[derive(Clone, Debug)]
pub struct SparseMat {
    n_rows: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    data: Vec<f64>,
}

impl SparseMat {
    pub fn new(n_rows: usize) -> SparseMat {
        SparseMat {
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

    pub fn push(&mut self, row: usize, val: f64) {
        self.indices.push(row);
        self.data.push(val);
    }

    pub fn seal_column(&mut self) {
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

    pub fn append_col<'a, T>(&mut self, col: T)
    where
        T: IntoIterator<Item = (usize, &'a f64)>,
    {
        assert_eq!(*self.indptr.last().unwrap(), self.indices.len()); // prev column is sealed
        for (idx, &val) in col {
            self.indices.push(idx);
            self.data.push(val);
        }
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

    pub fn transpose(&self) -> SparseMat {
        let mut out = SparseMat {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mat_transpose() {
        let mut mat = SparseMat::new(2);
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
}
