use super::sparse::Perm;

#[derive(Clone, Debug)]
struct Row {
    cols: Vec<usize>,
}

pub fn order_colamd<'a>(n_rows: usize, cols_iter: impl IntoIterator<Item = &'a [usize]>) -> Perm {
    // Implementation of (a part of) the COLAMD algorithm:
    //
    // "An approximate minimum degree column ordering algorithm",
    // S. I. Larimore, MS Thesis, Dept. of Computer and Information
    // Science and Engineering, University of Florida, Gainesville, FL,
    // 1998.  CISE Tech Report TR-98-016.
    //
    // https://www.researchgate.net/profile/Tim_Davis2/publication/220492488_A_column_approximate_minimum_degree_ordering_algorithm/links/551b1e100cf251c35b507fe5.pdf

    // TODO:
    // * allocate all storage at once
    // * remove dense rows
    // * immediately order dense columns
    // * deal with empty columns/rows
    // * supercolumns

    let mut rows = vec![Row { cols: vec![] }; n_rows];
    let mut cols = vec![];

    for (c, col_rows) in cols_iter.into_iter().enumerate() {
        for &r in col_rows {
            rows[r].cols.push(c);
        }

        cols.push(Col {
            rows: col_rows.to_vec(),
            score: 0,
        });
    }

    assert_eq!(rows.len(), cols.len());

    let max_score = cols.len() - 1;
    let mut cols_queue = ColsQueue::new(cols.len(), max_score);

    for c in 0..cols.len() {
        let col = &mut cols[c];

        let mut score = 0;
        for &r in &col.rows {
            score += rows[r].cols.len() - 1;
        }
        score = std::cmp::min(score, max_score);

        col.score = score;
        cols_queue.add(c, score);
    }

    // eprintln!(
    //     "INIT COLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
    //     cols, rows, col_scores
    // );

    let mut new2orig = Vec::with_capacity(cols.len());

    // cleared every iteration
    let mut is_in_pivot_row = vec![false; cols.len()];

    let mut is_absorbed_row = vec![false; n_rows];

    let mut row_set_diffs = vec![0; n_rows];
    let mut rows_with_diffs = vec![];
    let mut is_in_diffs = vec![false; n_rows];

    while new2orig.len() < cols.len() {
        let pivot_c = cols_queue.pop_min().unwrap();

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
                cols_queue.remove(c, cols[c].score);

                let mut diff = 0;
                for &r in &cols[c].rows {
                    diff += row_set_diffs[r];
                }

                if diff == 0 {
                    // mass elimination: we can order column c now as it will not result
                    // in any additional fill-in.
                    new2orig.push(c);
                    // eprintln!("ME: ORDERED {}", new2orig.last().unwrap());
                    pivot_row.swap_remove(i);
                    cols[c].rows.clear();
                } else {
                    cols[c].score = diff; // NOTE: not the final score, will be updated later.
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

                cols[c].score += pivot_row_len - 1; // TODO: supercolumns
                cols[c].score = std::cmp::min(cols[c].score, max_score);
                cols_queue.add(c, cols[c].score);
            }
        }

        // eprintln!(
        //     "AFTER ORDERING {} COLS:\nCOLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
        //     new2orig.len(), cols, rows, col_scores
        // );
    }

    let mut orig2new = vec![0; cols.len()];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    Perm { orig2new, new2orig }
}

#[derive(Clone, Debug)]
struct Col {
    rows: Vec<usize>,
    score: usize,
}

#[derive(Debug)]
struct ColsQueue {
    score2head: Vec<Option<usize>>,
    prev: Vec<usize>,
    next: Vec<usize>,
    min_score: usize,
}

impl ColsQueue {
    fn new(num_cols: usize, max_score: usize) -> ColsQueue {
        ColsQueue {
            score2head: vec![None; max_score + 1],
            prev: vec![0; num_cols],
            next: vec![0; num_cols],
            min_score: max_score + 1,
        }
    }

    fn pop_min(&mut self) -> Option<usize> {
        let col = loop {
            if self.min_score >= self.score2head.len() {
                return None;
            }
            if let Some(col) = self.score2head[self.min_score] {
                break col;
            }
            self.min_score += 1;
        };

        self.remove(col, self.min_score);
        Some(col)
    }

    fn add(&mut self, col: usize, score: usize) {
        self.min_score = std::cmp::min(self.min_score, score);

        if let Some(head) = self.score2head[score] {
            self.prev[col] = self.prev[head];
            self.next[col] = head;
            self.next[self.prev[head]] = col;
            self.prev[head] = col;
        } else {
            self.prev[col] = col;
            self.next[col] = col;
            self.score2head[score] = Some(col);
        }
    }

    fn remove(&mut self, col: usize, score: usize) {
        if self.next[col] == col {
            self.score2head[score] = None;
        } else {
            self.next[self.prev[col]] = self.next[col];
            self.prev[self.next[col]] = self.prev[col];
            if self.score2head[score].unwrap() == col {
                self.score2head[score] = Some(self.next[col]);
            }
        }
    }
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

        let perm = order_colamd(
            4,
            [0, 1, 2, 4]
                .iter()
                .map(|&c| mat.outer_view(c).unwrap().into_raw_storage().0),
        );
        assert_eq!(&perm.new2orig, &[1, 3, 0, 2]);
        assert_eq!(&perm.orig2new, &[2, 0, 3, 1]);
    }
}
