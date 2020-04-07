use sprs::{CsVecBase, CsVecView};
use std::ops::Deref;

pub(crate) fn resized_view<IStorage, DStorage>(
    vec: &CsVecBase<IStorage, DStorage>,
    len: usize,
) -> CsVecView<f64>
where
    IStorage: Deref<Target = [usize]>,
    DStorage: Deref<Target = [f64]>,
{
    let mut indices = vec.indices();
    let mut data = vec.data();
    while let Some(&i) = indices.last() {
        if i < len {
            // TODO: binary search
            break;
        }

        indices = &indices[..(indices.len() - 1)];
        data = &data[..(data.len() - 1)];
    }

    // Safety: new indices and data are the same size,indices are still sorted and all indices
    // are less than the new length. Thus, all CsVecView invariants are satisfied.
    unsafe { CsVecView::new_view_raw(len, data.len(), indices.as_ptr(), data.as_ptr()) }
}

pub(crate) fn to_dense<IStorage, DStorage>(vec: &CsVecBase<IStorage, DStorage>) -> Vec<f64>
where
    IStorage: Deref<Target = [usize]>,
    DStorage: Deref<Target = [f64]>,
{
    let mut dense = vec![0.0; vec.dim()];
    vec.scatter(&mut dense);
    dense
}

#[cfg(test)]
use sprs::{CsMat, CsVec};

#[cfg(test)]
pub(crate) fn to_sparse(slice: &[f64]) -> CsVec<f64> {
    let mut res = CsVec::empty(slice.len());
    for (i, &val) in slice.iter().enumerate() {
        if val != 0.0 {
            res.append(i, val);
        }
    }
    res
}

#[cfg(test)]
pub(crate) fn assert_matrix_eq(mat: &CsMat<f64>, reference: &[Vec<f64>]) {
    let mat = mat.to_csr();
    assert_eq!(mat.rows(), reference.len());
    for (r, row) in mat.outer_iterator().enumerate() {
        assert_eq!(to_dense(&row), reference[r], "matrices differ in row {}", r);
    }
}
