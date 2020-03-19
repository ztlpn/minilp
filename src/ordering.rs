use super::sparse::{Error, Perm};

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

#[allow(dead_code)]
pub fn order_colamd<'a>(
    size: usize,
    get_col: impl Fn(usize) -> &'a [usize],
) -> Result<Perm, Error> {
    // Implementation of (a part of) the COLAMD algorithm:
    //
    // "An approximate minimum degree column ordering algorithm",
    // S. I. Larimore, MS Thesis, Dept. of Computer and Information
    // Science and Engineering, University of Florida, Gainesville, FL,
    // 1998.  CISE Tech Report TR-98-016.
    //
    // https://www.researchgate.net/profile/Tim_Davis2/publication/220492488_A_column_approximate_minimum_degree_ordering_algorithm/links/551b1e100cf251c35b507fe5.pdf
    //
    // Additionally, we order columns of size 1 first (as they don't cause any fill-in). COLAMD
    // works best for *irreducible* matrices so ideally before ordering one should first reduce
    // the matrix to the block-triangular form and then apply LU factorization (and ordering)
    // to diagonal blocks. But it is more complicated and in LP most blocks are singletons anyway
    // so we limit ourselves to singleton columns.
    //
    // TODO:
    // * order empty columns/rows
    // * supercolumns

    let mut cols = vec![Slice { begin: 0, end: 0 }; size];
    let mut row_storage = vec![];

    let mut new2orig = vec![0; size];
    let mut cur_ordered_col = 0;
    let mut is_ordered_col = vec![false; size];

    let mut rows = vec![Slice { begin: 0, end: 0 }; size];
    let mut is_absorbed_row = vec![false; size];

    let mut num_cheap_singletons = 0;

    {
        // Gather columns and in the process cheaply order columns of size 1.
        for c in 0..size {
            let rows_begin = row_storage.len();
            for &r in get_col(c) {
                if !is_absorbed_row[r] {
                    row_storage.push(r);
                }
            }

            let rows_end = row_storage.len();
            if rows_end - rows_begin == 0 {
                return Err(Error::SingularMatrix);
            } else if rows_end - rows_begin > 1 {
                cols[c].begin = rows_begin;
                cols[c].end = rows_end;
                for &r in &row_storage[rows_begin..rows_end] {
                    rows[r].end += 1;
                }
            } else {
                num_cheap_singletons += 1;
                is_ordered_col[c] = true;
                new2orig[cur_ordered_col] = c;
                cur_ordered_col += 1;

                is_absorbed_row[row_storage[rows_begin]] = true;
                row_storage.pop();
            }
        }
    }

    for r in 0..size {
        if rows[r].end == 0 && !is_absorbed_row[r] {
            return Err(Error::SingularMatrix);
        }
    }

    let mut col_storage = vec![0; row_storage.len()];

    {
        // Gather rows.
        for i in 1..size {
            let prev_end = rows[i - 1].end;
            rows[i].begin = prev_end;
            rows[i].end += prev_end;
        }

        for c in 0..size {
            for &r in cols[c].elems(&row_storage) {
                let row = &mut rows[r];
                col_storage[row.begin] = c;
                row.begin += 1;
            }
        }

        rows[0].begin = 0;
        for i in 1..size {
            rows[i].begin = rows[i - 1].end;
        }
    }

    let mut num_singletons = num_cheap_singletons;

    {
        // Order remaining columns of size 1, taking into account that
        // eliminating some column can make other columns into singletons.
        let mut col_rows_len = cols.iter().map(|c| c.end - c.begin).collect::<Vec<_>>();
        let mut stack = vec![];
        for c in 0..size {
            if col_rows_len[c] != 1 {
                continue;
            }

            // all columns on the stack are of size 1
            stack.clear();
            stack.push(c);
            while !stack.is_empty() {
                let c = stack.pop().unwrap();
                let r = *cols[c]
                    .elems(&row_storage)
                    .iter()
                    .find(|&&r| !is_absorbed_row[r])
                    .unwrap();
                for &other_c in rows[r].elems(&col_storage) {
                    col_rows_len[other_c] -= 1;
                    if col_rows_len[other_c] == 1 {
                        stack.push(other_c);
                    }
                }

                is_absorbed_row[r] = true;

                num_singletons += 1;
                is_ordered_col[c] = true;
                new2orig[cur_ordered_col] = c;
                cur_ordered_col += 1;
            }
        }
    }

    let ns_size = size - num_singletons; // number of non-singleton columns.
    let mut num_dense_rows = 0;

    {
        // Dense rows make COLAMD bounds on fill-in useless so we exclude them from consideration
        // and hope that they won't be chosen during pivoting.
        let dense_row_thresh = std::cmp::max(16, ns_size / 4);
        for (r, row) in rows.iter_mut().enumerate() {
            if row.end - row.begin >= dense_row_thresh {
                is_absorbed_row[r] = true;
                num_dense_rows += 1;
            }
        }
    }

    let mut cols_queue = ColsQueue::new(cols.len());
    let mut num_dense_cols = 0;
    let mut num_cols_only_dense_rows = 0;

    {
        // Compact columns and detect dense ones.
        let dense_col_thresh = std::cmp::max(16, (ns_size as f64).sqrt() as usize);
        for c in 0..cols.len() {
            if is_ordered_col[c] {
                continue;
            }

            let col = &mut cols[c];
            let mut cur_end = col.begin;
            for i in col.begin..col.end {
                let r = row_storage[i];
                if !is_absorbed_row[r] {
                    row_storage[cur_end] = r;
                    cur_end += 1;
                }
            }

            col.end = cur_end;

            let col_len = cur_end - col.begin;
            if col_len >= dense_col_thresh {
                num_dense_cols += 1;
                cols_queue.add(c, col_len); // order dense columns by their size.
            } else if col_len == 0 {
                // This means the column consists only of dense rows.
                // Order these at the very end.
                num_cols_only_dense_rows += 1;
                cols_queue.add(c, size - 1);
            }
        }
    }

    {
        // order dense columns at the end.
        let cols_queue_len = cols_queue.len();
        for i in 0..cols_queue_len {
            let dense_c = cols_queue.pop_min().unwrap();
            new2orig[size - cols_queue_len + i] = dense_c;
            is_ordered_col[dense_c] = true;
        }
        assert_eq!(cols_queue.len(), 0);
    }

    let mut col_scores = vec![0; size];

    {
        // Calculate initial scores for sparse columns.
        // Use the same cols_queue (which was emptied on the previous step).
        for c in 0..cols.len() {
            if is_ordered_col[c] {
                continue;
            }

            let col = &mut cols[c];

            let mut score = 0;
            for &r in col.elems(&row_storage) {
                let row = &rows[r];
                score += row.end - row.begin - 1;
            }
            score = std::cmp::min(score, size - 1);

            col_scores[c] = score;
            cols_queue.add(c, score);
        }
    }

    // cleared every iteration
    let mut is_in_pivot_row = vec![false; cols.len()];

    let mut row_set_diffs = vec![0; size];
    let mut rows_with_diffs = vec![];
    let mut is_in_diffs = vec![false; size];

    let mut num_mass_eliminated = 0;

    while cols_queue.len() > 0 {
        let pivot_c = cols_queue.pop_min().unwrap();
        let pivot_row_begin = col_storage.len();

        new2orig[cur_ordered_col] = pivot_c;
        cur_ordered_col += 1;
        is_ordered_col[pivot_c] = true;

        let mut pivot_r = None;
        for &r in cols[pivot_c].elems(&row_storage) {
            if !std::mem::replace(&mut is_absorbed_row[r], true) {
                pivot_r = Some(r); // choose any absorbed row index to represent pivot row.
                let row = &rows[r];
                for i in row.begin..row.end {
                    let c = col_storage[i];
                    if !is_ordered_col[c] && !std::mem::replace(&mut is_in_pivot_row[c], true) {
                        col_storage.push(c);
                    }
                }
            }
        }
        let pivot_r = pivot_r.unwrap();

        // clear for next iteration.
        for &c in &col_storage[pivot_row_begin..] {
            is_in_pivot_row[c] = false;
        }

        // find row set differences
        for &c in &col_storage[pivot_row_begin..] {
            for &r in cols[c].elems(&row_storage) {
                if is_absorbed_row[r] {
                    continue;
                }

                if !is_in_diffs[r] {
                    is_in_diffs[r] = true;
                    rows_with_diffs.push(r);
                    let row = &rows[r];
                    row_set_diffs[r] = row.end - row.begin;
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
            let mut cur_pivot_i = pivot_row_begin;
            for pivot_i in pivot_row_begin..col_storage.len() {
                let c = col_storage[pivot_i];
                cols_queue.remove(c, col_scores[c]);

                let col = &mut cols[c];

                // calculate diff, compacting columns on the way
                let mut diff = 0;
                let mut cur_end = col.begin;
                for i in col.begin..col.end {
                    let r = row_storage[i];
                    if !is_absorbed_row[r] {
                        row_storage[cur_end] = r;
                        cur_end += 1;
                        diff += row_set_diffs[r];
                    }
                }
                col.end = cur_end;

                if diff == 0 {
                    // mass elimination: we can order column c now as it will not result
                    // in any additional fill-in.
                    num_mass_eliminated += 1;
                    new2orig[cur_ordered_col] = c;
                    cur_ordered_col += 1;
                    is_ordered_col[c] = true;
                } else {
                    col_scores[c] = diff; // NOTE: not the final score, will be updated later.
                    col_storage[cur_pivot_i] = c;
                    cur_pivot_i += 1;
                }
            }
            col_storage.truncate(cur_pivot_i);
        }

        // clear diffs for next iteration
        for &r in &rows_with_diffs {
            row_set_diffs[r] = 0;
            is_in_diffs[r] = false;
        }
        rows_with_diffs.clear();

        // TODO: detect supercolumns

        // calculate final column scores
        let pivot_row_len = col_storage.len() - pivot_row_begin;
        if pivot_row_len > 0 {
            let row = &mut rows[pivot_r];
            row.begin = pivot_row_begin;
            row.end = col_storage.len();
            is_absorbed_row[pivot_r] = false;
            for &c in row.elems(&col_storage) {
                let score = &mut col_scores[c];
                *score += pivot_row_len - 1; // TODO: supercolumns
                *score = std::cmp::min(*score, size - 1);
                cols_queue.add(c, *score);
            }
        }
    }

    let mut orig2new = vec![0; size];
    for (new, &orig) in new2orig.iter().enumerate() {
        orig2new[orig] = new;
    }

    trace!(
        "COLAMD: ordered {} cols, singletons: {} (cheap: {}), dense_rows: {}, dense_cols: {}, cols_only_dense_rows: {}, mass_eliminated: {}",
        size, num_singletons, num_cheap_singletons, num_dense_rows, num_dense_cols, num_cols_only_dense_rows, num_mass_eliminated);

    Ok(Perm { orig2new, new2orig })
}

#[derive(Clone, Debug)]
struct Slice {
    begin: usize,
    end: usize,
}

impl Slice {
    fn elems<'a>(&self, storage: &'a [usize]) -> &'a [usize] {
        &storage[self.begin..self.end]
    }
}

#[derive(Debug)]
struct ColsQueue {
    score2head: Vec<Option<usize>>,
    prev: Vec<usize>,
    next: Vec<usize>,
    min_score: usize,
    len: usize,
}

impl ColsQueue {
    fn new(num_cols: usize) -> ColsQueue {
        ColsQueue {
            score2head: vec![None; num_cols],
            prev: vec![0; num_cols],
            next: vec![0; num_cols],
            min_score: num_cols,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
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
        self.len += 1;

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
        self.len -= 1;
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

#[allow(dead_code)]
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

#[allow(dead_code)]
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
        })
        .unwrap();
        assert_eq!(&perm.new2orig, &[1, 0, 2, 3]);
        assert_eq!(&perm.orig2new, &[1, 0, 2, 3]);
    }

    #[test]
    fn colamd_singular() {
        {
            let empty_col_mat = mat_from_triplets(3, 3, &[(0, 0), (1, 0), (1, 1), (1, 2)]);
            let res = order_colamd(3, |c| {
                empty_col_mat.outer_view(c).unwrap().into_raw_storage().0
            });
            assert_eq!(res.unwrap_err(), Error::SingularMatrix);
        }

        {
            let empty_row_mat = mat_from_triplets(3, 3, &[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]);
            let res = order_colamd(3, |c| {
                empty_row_mat.outer_view(c).unwrap().into_raw_storage().0
            });
            assert_eq!(res.unwrap_err(), Error::SingularMatrix);
        }
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
