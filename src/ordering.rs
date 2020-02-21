use super::sparse::Perm;

pub fn order_colamd<'a>(size: usize, get_col: impl Fn(usize) -> &'a [usize]) -> Perm {
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

    let mut rows = vec![Row { cols: vec![] }; size];
    let mut cols = Vec::with_capacity(size);

    for c in 0..size {
        let col_rows = get_col(c);
        for &r in col_rows {
            rows[r].cols.push(c);
        }

        cols.push(Col {
            rows: col_rows.to_vec(),
            score: 0,
        });
    }

    assert_eq!(rows.len(), cols.len());

    let mut cols_queue = ColsQueue::new(cols.len());

    for c in 0..cols.len() {
        let col = &mut cols[c];

        let mut score = 0;
        for &r in &col.rows {
            score += rows[r].cols.len() - 1;
        }
        score = std::cmp::min(score, size - 1);

        col.score = score;
        cols_queue.add(c, score);
    }

    // eprintln!(
    //     "INIT COLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
    //     cols, rows, col_scores
    // );

    let mut new2orig = Vec::with_capacity(cols.len());

    let mut is_ordered_col = vec![false; size];
    let mut is_absorbed_row = vec![false; size];

    // cleared every iteration
    let mut pivot_row = vec![];
    let mut is_in_pivot_row = vec![false; cols.len()];

    let mut row_set_diffs = vec![0; size];
    let mut rows_with_diffs = vec![];
    let mut is_in_diffs = vec![false; size];

    while new2orig.len() < cols.len() {
        let pivot_c = cols_queue.pop_min().unwrap();

        new2orig.push(pivot_c);
        is_ordered_col[pivot_c] = true;
        // eprintln!("ORDERED {}", new2orig.last().unwrap());

        pivot_row.clear();
        let mut pivot_r = None;
        for &r in &cols[pivot_c].rows {
            if !std::mem::replace(&mut is_absorbed_row[r], true) {
                pivot_r = Some(r);  // choose any absorbed row index to represent pivot row.
                for &c in &rows[r].cols {
                    if !is_ordered_col[c] && !std::mem::replace(&mut is_in_pivot_row[c], true) {
                        pivot_row.push(c);
                    }
                }

                rows[r].cols.clear();
            }
        }
        let pivot_r = pivot_r.unwrap();

        // clear for next iteration.
        for &c in &pivot_row {
            is_in_pivot_row[c] = false;
        }

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
                    is_absorbed_row[r] = true;
                    rows[r].cols.clear();
                }
            }
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
                    is_ordered_col[c] = true;
                    pivot_row.swap_remove(i);
                    // eprintln!("ME: ORDERED {}", new2orig.last().unwrap());
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
            rows[pivot_r].cols = pivot_row.clone();
            is_absorbed_row[pivot_r] = false;
            for &c in &rows[pivot_r].cols {
                cols[c].score += pivot_row.len() - 1; // TODO: supercolumns
                cols[c].score = std::cmp::min(cols[c].score, size - 1);
                cols_queue.add(c, cols[c].score);
            }
        }

        // eprintln!(
        //     "AFTER ORDERING {} COLS:\nCOLS: {:?}\nROWS: {:?}\nSCORES: {:?}",
        //     new2orig.len(), cols, rows, col_scores
        // );
    }

    let mut orig2new = vec![0; size];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    Perm { orig2new, new2orig }
}

#[derive(Clone, Debug)]
struct Row {
    cols: Vec<usize>,
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
    fn new(num_cols: usize) -> ColsQueue {
        ColsQueue {
            score2head: vec![None; num_cols],
            prev: vec![0; num_cols],
            next: vec![0; num_cols],
            min_score: num_cols,
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

        let perm = order_colamd(4, |c| {
            mat.outer_view([0, 1, 2, 4][c])
                .unwrap()
                .into_raw_storage()
                .0
        });
        assert_eq!(&perm.new2orig, &[1, 3, 0, 2]);
        assert_eq!(&perm.orig2new, &[2, 0, 3, 1]);
    }
}
