use super::sparse::Perm;

/// Simplest preordering: order columns based on their size
pub fn order_simple<'a>(size: usize, get_col: impl Fn(usize) -> &'a [usize]) -> Perm {
    let mut cols_queue = ColsQueue::new(size);
    for c in 0..size {
        cols_queue.add(c, get_col(c).len() - 1);
    }

    let mut new2orig = Vec::with_capacity(size);
    while new2orig.len() < size {
        new2orig.push(cols_queue.pop_min().unwrap());
    }

    let mut orig2new = vec![0; size];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    Perm { orig2new, new2orig }
}

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

    let mut new2orig = Vec::with_capacity(cols.len());

    let mut col_rows_len = Vec::with_capacity(cols.len());
    for c in 0..size {
        let col_rows = get_col(c);
        for &r in col_rows {
            rows[r].cols.push(c);
        }

        cols.push(Col {
            rows: col_rows.to_vec(),
            score: 0,
        });

        col_rows_len.push(col_rows.len());
    }

    let mut is_ordered_col = vec![false; size];
    let mut is_absorbed_row = vec![false; size];

    // order columns of size 1
    let mut stack = vec![];
    for c in 0..size {
        if col_rows_len[c] == 0 || col_rows_len[c] > 1 {
            continue;
        }

        // all columns on the stack are of size 1
        stack.clear();
        stack.push(c);
        while !stack.is_empty() {
            let c = stack.pop().unwrap();
            let r = *cols[c].rows.iter().find(|&&r| !is_absorbed_row[r]).unwrap();
            for &other_c in &rows[r].cols {
                col_rows_len[other_c] -= 1;
                if col_rows_len[other_c] == 1 {
                    stack.push(other_c);
                }
            }

            rows[r].cols.clear();
            is_absorbed_row[r] = true;

            cols[c].rows.clear();
            is_ordered_col[c] = true;
            new2orig.push(c);
        }
    }

    // compact columns
    for c in 0..cols.len() {
        let col = &mut cols[c];
        let mut cur_i = 0;
        for i in 0..col.rows.len() {
            let r = col.rows[i];
            if !is_absorbed_row[r] {
                col.rows[cur_i] = r;
                cur_i += 1;
            }
        }
        assert_eq!(cur_i, col_rows_len[c]);
        if !is_ordered_col[c] {
            assert!(cur_i > 1);
        }
        col.rows.truncate(cur_i);
    }

    // calculate initial scores

    let mut cols_queue = ColsQueue::new(cols.len());

    for c in 0..cols.len() {
        if is_ordered_col[c] {
            continue;
        }

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
                pivot_r = Some(r); // choose any absorbed row index to represent pivot row.
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
                }
            }
        }

        // calculate row set differences with the pivot row
        {
            let mut cur_pivot_i = 0;
            for pivot_i in 0..pivot_row.len() {
                let c = pivot_row[pivot_i];
                cols_queue.remove(c, cols[c].score);

                // calculate diff, compacting columns on the way
                let mut diff = 0;
                let mut cur_i = 0;
                for i in 0..cols[c].rows.len() {
                    let r = cols[c].rows[i];
                    if !is_absorbed_row[r] {
                        cols[c].rows[cur_i] = r;
                        cur_i += 1;
                        diff += row_set_diffs[r];
                    }
                }

                if diff == 0 {
                    // mass elimination: we can order column c now as it will not result
                    // in any additional fill-in.
                    new2orig.push(c);
                    is_ordered_col[c] = true;
                    cols[c].rows.clear();
                // eprintln!("ME: ORDERED {}", new2orig.last().unwrap());
                } else {
                    cols[c].score = diff; // NOTE: not the final score, will be updated later.
                    cols[c].rows.truncate(cur_i);
                    pivot_row[cur_pivot_i] = c;
                    cur_pivot_i += 1;
                }
            }
            pivot_row.truncate(cur_pivot_i);
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
        if !pivot_row.is_empty() {
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

pub fn order_colamd_ffi<'a>(size: usize, get_col: impl Fn(usize) -> &'a [usize]) -> Perm {
    let mut new2orig = Vec::with_capacity(size);

    let mut is_singleton_row = vec![false; size];
    let mut nonsingleton_cols = vec![];
    for c in 0..size {
        let col_rows = get_col(c);
        if col_rows.len() == 1 {
            is_singleton_row[col_rows[0]] = true;
            new2orig.push(c);
        } else {
            nonsingleton_cols.push(c);
        }
    }

    let mut ns_row = 0;
    let mut row2ns_row = vec![0; size];
    for r in 0..size {
        if !is_singleton_row[r] {
            row2ns_row[r] = ns_row;
            ns_row += 1;
        }
    }

    let ns_size = nonsingleton_cols.len();

    if ns_size > 0 {
        let nnz = nonsingleton_cols
            .iter()
            .map(|&c| get_col(c).len())
            .sum::<usize>();
        let rec_len =
            unsafe { colamd_rs::colamd_recommended(nnz as i32, ns_size as i32, ns_size as i32) };

        let mut mat = vec![0i32; rec_len as usize];
        let mut mat_p = Vec::with_capacity(ns_size + 1);
        mat_p.push(0i32);
        for &c in &nonsingleton_cols {
            let col_rows = get_col(c);
            let begin = *mat_p.last().unwrap() as usize;
            let mut end = begin;
            for &r in col_rows {
                if !is_singleton_row[r] {
                    mat[end] = row2ns_row[r] as i32;
                    end += 1;
                }
            }
            mat_p.push(end as i32);
        }

        let mut knobs = vec![0.0; colamd_rs::COLAMD_KNOBS as usize];
        unsafe {
            colamd_rs::colamd_set_defaults(knobs.as_mut_ptr());
        }
        knobs[0] = 1.0;
        knobs[1] = 1.0;
        let mut stats = vec![0; colamd_rs::COLAMD_STATS as usize];

        let res = unsafe {
            colamd_rs::colamd(
                ns_size as i32,
                ns_size as i32,
                rec_len as i32,
                mat.as_mut_ptr(),
                mat_p.as_mut_ptr(),
                knobs.as_mut_ptr(),
                stats.as_mut_ptr(),
            )
        };
        assert_eq!(res, 1);

        unsafe {
            colamd_rs::colamd_report(stats.as_mut_ptr());
        }

        for &c in &mat_p[0..ns_size] {
            new2orig.push(nonsingleton_cols[c as usize]);
        }
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

const SENTINEL: usize = 0usize.wrapping_sub(1);

pub fn find_diag_matching<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> Option<Vec<usize>> {
    let mut col2visited_on_iter = vec![SENTINEL; size];
    let mut row2matched_col = vec![SENTINEL; size];
    // for each col a pointer to the position in its adjacency lists where we last looked for neighbors.
    let mut cheap = vec![0; size];

    struct Step {
        col: usize,
        cur_i: usize,
    }

    let mut dfs_stack = vec![];
    for start_c in 0..size {
        let mut found = false; // whether the dfs iteration found the match

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });

        'dfs_loop: while !dfs_stack.is_empty() {
            let mut cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            let col_rows = get_col(c);

            if col2visited_on_iter[c] != start_c {
                col2visited_on_iter[c] = start_c;

                let cur_cheap = &mut cheap[c];
                while *cur_cheap < col_rows.len() {
                    let r = col_rows[*cur_cheap];
                    if row2matched_col[r] == SENTINEL {
                        row2matched_col[r] = c;
                        found = true;
                        dfs_stack.pop();
                        continue 'dfs_loop;
                    }
                    *cur_cheap += 1;
                }
            } else {
                if found {
                    let r = col_rows[cur_step.cur_i];
                    row2matched_col[r] = c;
                    dfs_stack.pop();
                    continue 'dfs_loop;
                }

                cur_step.cur_i += 1;
            }

            while cur_step.cur_i < col_rows.len() {
                let r = col_rows[cur_step.cur_i];
                if col2visited_on_iter[row2matched_col[r]] != start_c {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i == col_rows.len() {
                dfs_stack.pop();
            } else {
                let col = row2matched_col[col_rows[cur_step.cur_i]];
                dfs_stack.push(Step { col, cur_i: 0 });
            }
        }

        if !found {
            return None;
        }
    }

    Some(row2matched_col)
}

/// Lower block triangular form of a matrix.
#[derive(Clone, Debug)]
pub struct BlockDiagForm {
    /// Row permutation: for each original row its new row number so that diag is nonzero.
    pub row2col: Vec<usize>,
    /// For each block its set of columns (the order of blocks is lower block triangular)
    pub block_cols: Vec<Vec<usize>>,
}

pub fn find_block_diag_form<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> BlockDiagForm {
    let row2col = find_diag_matching(size, &get_col).unwrap();

    struct Step {
        col: usize,
        cur_i: usize,
    }

    let mut dfs_stack = vec![];
    let mut visited = vec![];
    let mut is_visited = vec![false; size];
    for start_c in 0..size {
        if is_visited[start_c] {
            continue;
        }

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });
        while !dfs_stack.is_empty() {
            let cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            if !is_visited[c] {
                is_visited[c] = true;
            } else {
                cur_step.cur_i += 1;
            }

            let col_rows = get_col(c);
            while cur_step.cur_i < col_rows.len() {
                let next_c = row2col[col_rows[cur_step.cur_i]];
                if !is_visited[next_c] {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i < col_rows.len() {
                let col = row2col[col_rows[cur_step.cur_i]];
                dfs_stack.push(Step { col, cur_i: 0 });
            } else {
                visited.push(c);
                dfs_stack.pop();
            }
        }
    }

    // Prepare transposed graph
    // TODO: more efficient transpose without allocating each row.
    let mut rows = vec![vec![]; size];
    for c in 0..size {
        for &r in get_col(c) {
            rows[row2col[r]].push(c);
        }
    }

    is_visited.clear();
    is_visited.resize(size, false);

    let mut block_cols = vec![];

    // DFS on the transposed graph
    for &start_c in visited.iter().rev() {
        if is_visited[start_c] {
            continue;
        }

        block_cols.push(vec![]);

        dfs_stack.clear();
        dfs_stack.push(Step {
            col: start_c,
            cur_i: 0,
        });
        while !dfs_stack.is_empty() {
            let cur_step = dfs_stack.last_mut().unwrap();
            let c = cur_step.col;
            if !is_visited[c] {
                is_visited[c] = true;
                block_cols.last_mut().unwrap().push(c);
            } else {
                cur_step.cur_i += 1;
            }

            let next = &rows[c];
            while cur_step.cur_i < next.len() {
                if !is_visited[next[cur_step.cur_i]] {
                    break;
                }
                cur_step.cur_i += 1;
            }

            if cur_step.cur_i < next.len() {
                let col = next[cur_step.cur_i];
                dfs_stack.push(Step { col, cur_i: 0 });
            } else {
                dfs_stack.pop();
            }
        }
    }

    BlockDiagForm {
        row2col,
        block_cols,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sprs::{CsMat, TriMat};

    fn mat_from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize)]) -> CsMat<f64> {
        let mut mat = TriMat::with_capacity((rows, cols), triplets.len());
        for (r, c) in triplets {
            mat.add_triplet(*r, *c, 1.0);
        }
        mat.to_csc()
    }

    #[test]
    fn colamd() {
        let mat = mat_from_triplets(
            4,
            5,
            &[
                (0, 0),
                (0, 2),
                (1, 0),
                (1, 2),
                (1, 4),
                (2, 0),
                (2, 1),
                (2, 4),
                (3, 0),
                (3, 4),
            ],
        );

        let perm = order_colamd(4, |c| {
            mat.outer_view([0, 1, 2, 4][c])
                .unwrap()
                .into_raw_storage()
                .0
        });
        assert_eq!(&perm.new2orig, &[1, 0, 2, 3]);
        assert_eq!(&perm.orig2new, &[1, 0, 2, 3]);

        let perm = order_colamd_ffi(4, |c| {
            mat.outer_view([0, 1, 2, 4][c])
                .unwrap()
                .into_raw_storage()
                .0
        });
        assert_eq!(&perm.new2orig, &[1, 0, 2, 3]);
    }

    #[test]
    fn diag_matching() {
        let size = 3;
        let mat = mat_from_triplets(
            size,
            size,
            &[(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0)],
        );

        let matching =
            find_diag_matching(size, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
        assert_eq!(matching, Some(vec![1, 2, 0]));
    }

    #[test]
    fn block_diag_form() {
        let size = 3;
        let mat = mat_from_triplets(
            size,
            size,
            &[(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)],
        );

        let bd_form =
            find_block_diag_form(size, |c| mat.outer_view(c).unwrap().into_raw_storage().0);
        assert_eq!(bd_form.row2col, &[2, 0, 1]);
        assert_eq!(bd_form.block_cols, vec![vec![0, 1], vec![2]]);
    }
}
