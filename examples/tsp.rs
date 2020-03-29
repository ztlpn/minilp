#[macro_use]
extern crate log;

use minilp::{LinearExpr, OptimizationDirection, RelOp, Variable};

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn sqr_dist(self, other: Self) -> f64 {
        (self.x - other.x) * (self.x - other.x) + (self.y - other.y) * (self.y - other.y)
    }

    fn dist(self, other: Self) -> f64 {
        f64::sqrt(self.sqr_dist(other))
    }
}

struct Problem {
    points: Vec<Point>,
}

fn read_line() -> Vec<String> {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
    line.split_whitespace().map(|tok| tok.to_owned()).collect()
}

impl Problem {
    fn read() -> Problem {
        let num_points = read_line()[0].parse::<usize>().unwrap();
        let mut points = vec![];
        for _ in 0..num_points {
            let line = read_line();
            points.push(Point {
                x: line[0].parse::<f64>().unwrap(),
                y: line[1].parse::<f64>().unwrap(),
            });
        }
        Problem { points }
    }

    fn dist(&self, n1: usize, n2: usize) -> f64 {
        self.points[n1].dist(self.points[n2])
    }

    fn solve(&self) {
        let num_points = self.points.len();
        let mut lp_problem = minilp::Problem::new(OptimizationDirection::Minimize);

        let mut edge_vars = vec![vec![]; num_points];
        for i in 0..num_points {
            for j in 0..i {
                edge_vars[i].push(lp_problem.add_var(Some(0.0), Some(1.0), self.dist(i, j)));
            }
        }

        let get_edge_var = |i: usize, j: usize| -> Variable {
            assert_ne!(i, j);
            if i > j {
                edge_vars[i][j]
            } else {
                edge_vars[j][i]
            }
        };

        for i in 0..num_points {
            let mut edges_sum = LinearExpr::empty();
            for j in 0..num_points {
                if i != j {
                    edges_sum.add(get_edge_var(i, j), 1.0);
                }
            }
            lp_problem.add_constraint(edges_sum, RelOp::Eq, 2.0);
        }

        let mut cur_solution = lp_problem.solve().unwrap();
        info!(
            "solved initial LP problem, objective: {}",
            cur_solution.objective()
        );
        cur_solution = add_subtour_constraints(cur_solution, &edge_vars);
        info!(
            "objective after adding initial subtour constraints: {}",
            cur_solution.objective()
        );
    }
}

/// Add all subtour constraints violated by the current solution.
/// A subtour constraint states that the sum of edge values for all edges going out
/// of some proper subset of nodes must be >= 2. This prevents the formation of closed subtours
/// that do not go through all the vertices.
fn add_subtour_constraints(
    mut cur_solution: minilp::Solution,
    edge_vars: &[Vec<Variable>],
) -> minilp::Solution {
    let num_points = edge_vars.len();
    let mut edge_weights = Vec::with_capacity(num_points * num_points);
    loop {
        edge_weights.clear();
        edge_weights.resize(num_points * num_points, 0.0);
        for i in 0..num_points {
            for j in 0..num_points {
                let edge_var = if i > j {
                    edge_vars[i][j]
                } else if i < j {
                    edge_vars[j][i]
                } else {
                    continue;
                };

                edge_weights[i * num_points + j] = cur_solution[edge_var];
            }
        }

        let (cut_weight, cut_mask) = find_min_cut(num_points, &mut edge_weights);
        if cut_weight > 2.0 - 1e-8 {
            // If the weight of min cut is >= 2 then no subtour constraints are violated.
            return cur_solution;
        }

        let mut cut_edges_sum = LinearExpr::empty();
        for i in 0..num_points {
            for j in 0..i {
                if cut_mask[i] != cut_mask[j] {
                    cut_edges_sum.add(edge_vars[i][j], 1.0);
                }
            }
        }

        cur_solution = cur_solution
            .add_constraint(cut_edges_sum, RelOp::Ge, 2.0)
            .unwrap();
    }
}

/// Given an undirected graph with weighted edges, find a cut (a partition of the graph nodes
/// into two non-empty sets) with the minimum weight (sum of weights of edges that go between
/// the two sets of the cut).
///
/// Input parameters: size is the number of nodes and weights is the flattened representation of
/// the square weights matrix.
///
/// Returns the minimum cut weight and the cut itself, represented by the boolean mask
/// (cut[node] is true if the node is in one set and false if it is in another).
fn find_min_cut(size: usize, weights: &mut [f64]) -> (f64, Vec<bool>) {
    // https://en.wikipedia.org/wiki/Stoer–Wagner_algorithm

    assert!(size >= 2);
    assert_eq!(weights.len(), size * size);

    let mut is_merged = vec![false; size];
    // Clusters are represented by linked lists. Each node points to the next in cluster except
    // the last one which points to itself. For all nodes except heads of lists is_merged is true.
    let mut next_in_cluster = (0..size).collect::<Vec<_>>();

    // True if node was added to the growing graph subset during the phase.
    // Cleared every phase.
    let mut is_added = Vec::with_capacity(size);

    // Sum of all edge weights for edges to the currently added part of the graph.
    // Cleared every phase.
    let mut node_weights = Vec::with_capacity(size);

    let mut best_cut_weight = f64::INFINITY;
    // Boolean mask for the best cut.
    let mut best_cut = Vec::with_capacity(size);

    for i_phase in 0..(size - 1) {
        is_added.clear();
        is_added.extend_from_slice(&is_merged);

        // Initially the subset is just the node 0 and node weights are weights of edges from
        // 0 to that nodes.
        node_weights.clear();
        node_weights.extend_from_slice(&weights[0..size]);

        let mut prev_node = 0;
        let mut last_node = 0;
        for _ in 1..(size - i_phase) {
            prev_node = last_node;
            let mut max_weight = f64::NEG_INFINITY;
            for n in 1..size {
                if !is_added[n] && node_weights[n] > max_weight {
                    last_node = n;
                    max_weight = node_weights[n];
                }
            }

            // last_node is the node with the biggest weight. Add it to the current subset
            // and update weights of other nodes to include edges that went from last_node to them.
            is_added[last_node] = true;
            for i in 0..size {
                if !is_added[i] {
                    node_weights[i] += weights[i * size + last_node];
                }
            }
        }

        // "Cut-of-the-phase" is the cut between the last_node cluster and the rest of the graph.
        // Its weight is the node weight of the last node.
        let cut_weight = node_weights[last_node];
        let is_best_cut = cut_weight < best_cut_weight;
        if is_best_cut {
            best_cut_weight = cut_weight;
            best_cut.clear();
            best_cut.resize(size, false);
        }

        // Merge the prev_node cluster into the last_node cluster. If the cut is best so far,
        // set the best_cut mask as we traverse the linked list.
        is_merged[prev_node] = true;
        let mut list_elem = last_node;
        loop {
            if is_best_cut {
                best_cut[list_elem] = true;
            }

            if next_in_cluster[list_elem] != list_elem {
                list_elem = next_in_cluster[list_elem];
            } else {
                next_in_cluster[list_elem] = prev_node;
                break;
            }
        }

        // Merge the weights of the prev_node edges into the weights of last_node edges.
        for n in 0..size {
            weights[last_node * size + n] += weights[prev_node * size + n];
        }
        for n in 0..size {
            weights[n * size + last_node] = weights[last_node * size + n];
        }
    }

    assert!(best_cut_weight.is_finite());
    return (best_cut_weight, best_cut);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_min_cut() {
        // Example from the original article with nodes 0 and 1 swapped.
        let size = 8;
        let weights = [
            [0.0, 2.0, 3.0, 0.0, 2.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 4.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 2.0, 2.0],
            [2.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0],
        ];
        let mut weights_flat = vec![];
        for row in &weights {
            weights_flat.extend_from_slice(row);
        }

        let (weight, cut_mask) = super::find_min_cut(size, &mut weights_flat);
        let nodes = (0..size).filter(|&n| cut_mask[n]).collect::<Vec<_>>();
        assert_eq!(weight, 4.0);
        assert_eq!(&nodes, &[2, 3, 6, 7]);
    }
}

fn main() {
    env_logger::init();

    let problem = Problem::read();
    problem.solve();
}
