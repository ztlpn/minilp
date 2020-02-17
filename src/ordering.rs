use super::sparse::Perm;
use sprs::CsMat;
use std::collections::BTreeSet;

#[derive(Clone, Debug)]
struct Row {
    cols: Vec<usize>,
}

#[derive(Clone, Debug)]
struct Col {
    rows: Vec<usize>,
}

pub fn order_colamd(mat: &CsMat<f64>, mat_cols: &[usize]) -> Perm {
    assert!(mat.is_csc());
    assert_eq!(mat.rows(), mat_cols.len());

    let mut rows = vec![Row { cols: vec![] }; mat.rows()];
    let mut cols = vec![Col { rows: vec![] }; mat_cols.len()];

    for c in 0..cols.len() {
        let col = mat.outer_view(mat_cols[c]).unwrap();
        cols[c].rows.extend_from_slice(col.indices());
        for &r in col.indices() {
            rows[r].cols.push(c);
        }
    }

    // TODO:
    // * better priority queue structure
    // * allocate all storage at once
    // * remove dense rows
    // * immediately order dense columns
    // * deal with empty columns/rows
    // * supercolumns

    let mut col_scores = vec![];
    let mut score_to_col = BTreeSet::<(usize, usize)>::new();
    for (c, col) in cols.iter().enumerate() {
        let mut score = 0;
        for &r in &col.rows {
            score += rows[r].cols.len() - 1;
        }
        col_scores.push(score);
        score_to_col.insert((score, c));
    }

    // eprintln!(
    //     "INIT COLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
    //     cols, rows, col_scores
    // );

    let mut new2orig = Vec::with_capacity(mat_cols.len());

    // cleared every iteration
    let mut is_in_pivot_row = vec![false; mat_cols.len()];

    let mut is_absorbed_row = vec![false; mat.rows()];

    let mut row_set_diffs = vec![0; mat.rows()];
    let mut rows_with_diffs = vec![];
    let mut is_in_diffs = vec![false; mat.rows()];

    while new2orig.len() < mat_cols.len() {
        let pivot_c = {
            let first = score_to_col.iter().next().unwrap().clone();
            score_to_col.take(&first).unwrap().1
        };

        new2orig.push(pivot_c);
        // eprintln!("ORDERED {}", new2orig.last().unwrap());

        for &r in &cols[pivot_c].rows {
            is_absorbed_row[r] = true;
        }

        let mut pivot_row = {
            let mut res = vec![];
            for &r in &cols[pivot_c].rows {
                for &c in &rows[r].cols {
                    if c != pivot_c && !is_in_pivot_row[c] {
                        is_in_pivot_row[c] = true;
                        res.push(c);
                    }
                }

                rows[r].cols.clear();
            }

            // clear for next iteration.
            for &c in &res {
                is_in_pivot_row[c] = false;
            }

            res
        };

        let mut absorbed_rows = std::mem::take(&mut cols[pivot_c].rows);

        // find row set differences
        for &c in &pivot_row {
            for &r in &cols[c].rows {
                if is_absorbed_row[r] {
                    continue;
                }

                if !is_in_diffs[r] {
                    is_in_diffs[r] = true;
                    rows_with_diffs.push(r);
                    row_set_diffs[r] = rows[r].cols.len();
                }

                row_set_diffs[r] -= 1; // TODO: supercolumns

                if row_set_diffs[r] == 0 {
                    // aggressive absorption
                    absorbed_rows.push(r);
                    is_absorbed_row[r] = true;
                    rows[r].cols.clear();
                }
            }
        }

        // remove absorbed rows from columns
        for &c in &pivot_row {
            let mut i = 0;
            while i < cols[c].rows.len() {
                if is_absorbed_row[cols[c].rows[i]] {
                    cols[c].rows.swap_remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // clear is_absorbed_row for next iteration
        for &r in &absorbed_rows {
            is_absorbed_row[r] = false;
        }

        // calculate row set differences with the pivot row
        {
            let mut i = 0;
            while i < pivot_row.len() {
                let c = pivot_row[i];

                let mut diff = 0;
                for &r in &cols[c].rows {
                    diff += row_set_diffs[r];
                }

                score_to_col.take(&(col_scores[c], c)).unwrap();
                if diff == 0 {
                    // mass elimination: we can order column c now as it will not result
                    // in any additional fill-in.
                    new2orig.push(c);
                    // eprintln!("ME: ORDERED {}", new2orig.last().unwrap());
                    pivot_row.swap_remove(i);
                    cols[c].rows.clear();
                } else {
                    col_scores[c] = diff; // NOTE: not the final score, will be updated later.
                    i += 1;
                }
            }
        }

        // eprintln!("PIVOT ROW {:?}", pivot_row);

        // clear diffs for next iteration
        for &r in &rows_with_diffs {
            row_set_diffs[r] = 0;
            is_in_diffs[r] = false;
        }
        rows_with_diffs.clear();

        // TODO: detect supercolumns

        // calculate final column scores
        {
            let pivot_r = *absorbed_rows.first().unwrap(); // choose any absorbed row index to represent pivot row.
            let pivot_row_len = pivot_row.len();
            rows[pivot_r].cols = pivot_row;
            for &c in &rows[pivot_r].cols {
                cols[c].rows.push(pivot_r);
                col_scores[c] += pivot_row_len - 1; // TODO: supercolumns
                score_to_col.insert((col_scores[c], c));
            }
        }

        // eprintln!(
        //     "AFTER ORDERING {} COLS:\nCOLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
        //     new2orig.len(), cols, rows, col_scores
        // );
    }

    let mut orig2new = vec![0; mat_cols.len()];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    Perm { orig2new, new2orig }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::TriMat;

    #[test]
    fn simple() {
        let triplets = [
            (0, 0, 1.0),
            (0, 2, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (1, 4, 1.0),
            (2, 0, 1.0),
            (2, 1, 1.0),
            (2, 4, 1.0),
            (3, 0, 1.0),
            (3, 4, 1.0),
        ];
        let mut mat = TriMat::with_capacity((4, 5), triplets.len());
        for (r, c, val) in &triplets {
            mat.add_triplet(*r, *c, *val);
        }
        let mat = mat.to_csc();

        let perm = order_colamd(&mat, &[0, 1, 2, 4]);
        assert_eq!(&perm.new2orig, &[1, 3, 0, 2]);
        assert_eq!(&perm.orig2new, &[2, 0, 3, 1]);
    }
}
