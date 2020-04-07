//! A solver for the travelling salesman problem.
//!
//! Solves euclidean TPS problems using the integer linear programming approach.
//! See comments in the solve() function for the detailed description of the algorithm.

#[macro_use]
extern crate log;

use minilp::{ComparisonOp, LinearExpr, OptimizationDirection, Variable};
use std::io;

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
    name: String,
    points: Vec<Point>,
}

fn read_line<R: io::BufRead>(mut input: R) -> io::Result<Vec<String>> {
    let mut line = String::new();
    input.read_line(&mut line)?;
    Ok(line.split_whitespace().map(|tok| tok.to_owned()).collect())
}

fn parse_num<T: std::str::FromStr>(input: &str, line_num: usize) -> io::Result<T> {
    input.parse::<T>().or(Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!("line {}: couldn't parse number", line_num),
    )))
}

impl Problem {
    fn dist(&self, n1: usize, n2: usize) -> f64 {
        self.points[n1].dist(self.points[n2])
    }

    /// Parse a problem in the TSPLIB format.
    ///
    /// Format description: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    fn parse<R: io::BufRead>(mut input: R) -> io::Result<Problem> {
        let mut name = String::new();
        let mut dimension = None;
        let mut line_num = 0;
        loop {
            let line = read_line(&mut input)?;
            if line.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "premature end of header".to_string(),
                ));
            }

            let mut keyword = line[0].clone();
            if keyword.ends_with(":") {
                keyword.pop();
            }

            if keyword == "NAME" {
                name = line.last().unwrap().clone();
            } else if keyword == "TYPE" {
                if line.last().unwrap() != "TSP" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "only problems with TYPE: TSP supported".to_string(),
                    ));
                }
            } else if keyword == "EDGE_WEIGHT_TYPE" {
                if line.last().unwrap() != "EUC_2D" {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "only problems with EDGE_WEIGHT_TYPE: EUC_2D supported".to_string(),
                    ));
                }
            } else if keyword == "DIMENSION" {
                let dim: usize = parse_num(line.last().as_ref().unwrap(), line_num)?;
                dimension = Some(dim);
            } else if keyword == "NODE_COORD_SECTION" {
                break;
            }

            line_num += 1;
        }

        let num_points = dimension.ok_or(io::Error::new(
            io::ErrorKind::InvalidData,
            "no DIMENSION specified".to_string(),
        ))?;
        if num_points > 100_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("problem dimension: {} is suspiciously large", num_points),
            ));
        }

        let mut point_opts = vec![None; num_points];
        for _ in 0..num_points {
            let line = read_line(&mut input)?;
            let node_num: usize = parse_num(&line[0], line_num)?;
            let x: f64 = parse_num(&line[1], line_num)?;
            let y: f64 = parse_num(&line[2], line_num)?;
            if node_num == 0 || node_num > num_points {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: bad node number: {}", line_num, node_num),
                ));
            }
            if point_opts[node_num - 1].is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("line {}: node {} specified twice", line_num, node_num),
                ));
            }
            point_opts[node_num - 1] = Some(Point { x, y });

            line_num += 1;
        }

        let line = read_line(input)?;
        if line.len() > 1 || (line.len() == 1 && line[0] != "EOF") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: expected EOF", line_num),
            ));
        }

        let mut points = vec![];
        for (i, po) in point_opts.into_iter().enumerate() {
            if let Some(p) = po {
                points.push(p);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("node {} is not specified", i),
                ));
            }
        }

        Ok(Problem { name, points })
    }
}

/// A solution to the TSP problem: a sequence of node indices in the tour order.
/// Each node must be present in the tour exactly once.
struct Tour(Vec<usize>);

impl Tour {
    fn to_string(&self) -> String {
        self.0
            .iter()
            .map(|n| (n + 1).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn to_svg(&self, problem: &Problem) -> String {
        let cmp_f64 = |x: &f64, y: &f64| x.partial_cmp(y).unwrap();
        let min_x = problem.points.iter().map(|p| p.x).min_by(cmp_f64).unwrap();
        let max_x = problem.points.iter().map(|p| p.x).max_by(cmp_f64).unwrap();
        let min_y = problem.points.iter().map(|p| p.y).min_by(cmp_f64).unwrap();
        let max_y = problem.points.iter().map(|p| p.y).max_by(cmp_f64).unwrap();

        let width = 600;
        let margin = 50;
        let scale = ((width - 2 * margin) as f64) / (max_x - min_x);
        let height = f64::round((max_y - min_y) * scale) as usize + 2 * margin;

        let mut svg = String::new();
        svg += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
        svg += "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n";
        svg += "  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n";
        svg += &format!(
            "<svg width=\"{}px\" height=\"{}px\" version=\"1.1\"",
            width, height
        );
        svg += "     xmlns=\"http://www.w3.org/2000/svg\">\n";

        use std::fmt::Write;
        svg += "    <path fill=\"none\" stroke=\"black\" stroke-width=\"4px\" d=\"\n";
        for &i in &self.0 {
            let p = problem.points[i];
            let px = f64::round((p.x - min_x) * scale) as usize + margin;
            let py = f64::round((p.y - min_y) * scale) as usize + margin;
            if i == 0 {
                writeln!(&mut svg, "        M {} {}", px, py).unwrap();
            } else {
                writeln!(&mut svg, "        L {} {}", px, py).unwrap();
            }
        }
        svg += "        Z\n";
        svg += "    \"/>\n";

        svg += "</svg>\n";
        svg
    }
}

fn solve(problem: &Problem) -> Tour {
    info!("starting, problem name: {}", problem.name);

    let num_points = problem.points.len();

    // First, we construct a linear programming model for the TSP problem.
    let mut lp_problem = minilp::Problem::new(OptimizationDirection::Minimize);

    // Variables in our model correspond to edges between nodes (cities). If the tour includes
    // the edge between nodes i and j, then the edge_vars[i][j] variable will be equal to 1.0 in
    // the solution. As we will solve the continuous version of the problem first, we constrain
    // each variable to the interval between 0.0 and 1.0 and deal with the fact that we want an
    // integer solution later. Each edge will contribute the distance between its endpoints to
    // the objective function which we want to minimize.
    let mut edge_vars = vec![vec![]; num_points];
    for i in 0..num_points {
        for j in 0..num_points {
            let var = if j < i {
                edge_vars[j][i]
            } else {
                lp_problem.add_var(problem.dist(i, j), (0.0, 1.0))
            };
            edge_vars[i].push(var);
        }
    }

    // Next, we add constraints that will ensure that each node is part of a tour.
    // To do this we specify that for each node exactly two edges incident on that node
    // are present in the solution. Or, equivalently: for every node the sum of all edge
    // variables incident on that node is equal to 2.0.
    for i in 0..num_points {
        let mut edges_sum = LinearExpr::empty();
        for j in 0..num_points {
            if i != j {
                edges_sum.add(edge_vars[i][j], 1.0);
            }
        }
        lp_problem.add_constraint(edges_sum, ComparisonOp::Eq, 2.0);
    }

    let mut cur_solution = lp_problem.solve().unwrap();

    // Even if an integer solution to the above problem is found, it is not enough. The problem
    // is that nothing in the model prohibits *subtours* - that is, tours that pass only
    // through a subset of all nodes. Unfortunately, to prohibit all subtours we must add
    // exponentially many constraints which is clearly infeasible to do beforehand. Instead,
    // we add these constraints *dynamically* - given the solution, add enough constraints
    // so that there are no subtours in that solution.
    cur_solution = add_subtour_constraints(cur_solution, &edge_vars);

    // Now we've got a solution to the continuous problem. This problem is called a *relaxation* -
    // integrality constraints are relaxed, but the integer solution that we want to find is still
    // somewhere in the feasible region of this problem (and its objective value is necessarily
    // higher). If the optimal solution is by chance integer, then we are done! Else we will try
    // to fix some variables in the solution to specific integer values and see if we can get an
    // integral solution this way. If we are able to find an integer solution and prove that no
    // better integer solution exists, we are done. This process is called *branch&bound*.

    // We explore the space of possible variable values using the depth-first search.
    // Struct Step represents an item in the DFS stack. We will choose a variable and
    // try to fix its value to either 0 or 1. After we explore a branch where one value
    // is chosen, we return and try another value.
    struct Step {
        start_solution: minilp::Solution, // LP solution right before the step.
        var: Variable,
        start_val: u8,
        cur_val: Option<u8>,
    }

    // As we want to get to high-quality solutions as quickly as possible, choice of the next
    // variable to fix and its initial value is important. This is necessarily a heuristic choice
    // and we use a simple heuristic: the next variable is the "most fractional" one and its
    // initial value is the closest integer to the current solution value.

    // Returns None if the solution is integral.
    fn choose_branch_var(cur_solution: &minilp::Solution) -> Option<Variable> {
        let mut max_divergence = 0.0;
        let mut max_var = None;
        for (var, &val) in cur_solution {
            let divergence = f64::abs(val - val.round());
            if divergence > 1e-5 && divergence > max_divergence {
                max_divergence = divergence;
                max_var = Some(var);
            }
        }
        max_var
    }

    fn new_step(start_solution: minilp::Solution, var: Variable) -> Step {
        let start_val = if start_solution[var] < 0.5 { 0 } else { 1 };
        Step {
            start_solution,
            var,
            start_val,
            cur_val: None,
        }
    }

    // We will save the best solution that we encountered in the search so far and its cost.
    // After we finish the search this will be the optimal tour.
    let mut best_cost = f64::INFINITY;
    let mut best_tour = None;

    let mut dfs_stack = if let Some(var) = choose_branch_var(&cur_solution) {
        info!(
            "starting branch&bound, current obj. value: {:.2}",
            cur_solution.objective(),
        );

        vec![new_step(cur_solution, var)]
    } else {
        info!(
            "found optimal solution with initial relaxation! cost: {:.2}",
            cur_solution.objective(),
        );

        return tour_from_lp_solution(&cur_solution, &edge_vars);
    };

    for iter in 0.. {
        let cur_step = dfs_stack.last_mut().unwrap();

        // Choose the next value for the current variable.
        if let Some(ref mut val) = cur_step.cur_val {
            if *val == cur_step.start_val {
                *val = 1 - *val;
            } else {
                // We've expored all values for the currrent variable so we must backtrack to
                // the previous step. If the stack becomes empty then our search is done.
                dfs_stack.pop();
                if dfs_stack.is_empty() {
                    break;
                } else {
                    continue;
                }
            }
        } else {
            cur_step.cur_val = Some(cur_step.start_val);
        };

        let mut cur_solution = cur_step.start_solution.clone();
        if let Ok(new_solution) =
            cur_solution.fix_var(cur_step.var, cur_step.cur_val.unwrap() as f64)
        {
            cur_solution = new_solution;
        } else {
            // There is no feasible solution with the current variable constraints.
            // We must backtrack.
            continue;
        }

        cur_solution = add_subtour_constraints(cur_solution, &edge_vars);

        let obj_val = cur_solution.objective();
        if obj_val > best_cost {
            // As the cost of any solution that we can find in the current branch is bound
            // from below by obj_val, it is pointless to explore this branch: we won't find
            // better solutions there.
            continue;
        }

        if let Some(var) = choose_branch_var(&cur_solution) {
            // Search deeper.
            dfs_stack.push(new_step(cur_solution, var));
        } else {
            // We've found an integral solution!
            if obj_val < best_cost {
                info!(
                    "iter {} (search depth {}): found new best solution, cost: {:.2}",
                    iter,
                    dfs_stack.len(),
                    obj_val
                );
                best_cost = obj_val;
                best_tour = Some(tour_from_lp_solution(&cur_solution, &edge_vars));
            }
        };
    }

    info!("found optimal solution, cost: {:.2}", best_cost);
    best_tour.unwrap()
}

/// Add all subtour constraints violated by the current solution.
/// A subtour constraint states that the sum of edge values for all edges going out
/// of some proper subset of nodes must be >= 2. This prevents the formation of closed subtours
/// that do not pass through all the vertices.
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
                if i != j {
                    edge_weights[i * num_points + j] = cur_solution[edge_vars[i][j]];
                }
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
            .add_constraint(cut_edges_sum, ComparisonOp::Ge, 2.0)
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

/// Convert a solution to the LP problem to the corresponding tour (a sequence of nodes).
/// Precondition: the solution must be integral and contain a unique tour.
fn tour_from_lp_solution(lp_solution: &minilp::Solution, edge_vars: &[Vec<Variable>]) -> Tour {
    let num_points = edge_vars.len();
    let mut tour = vec![];
    let mut is_visited = vec![false; num_points];
    let mut cur_point = 0;
    for _ in 0..num_points {
        assert!(!is_visited[cur_point]);
        is_visited[cur_point] = true;
        tour.push(cur_point);
        for neighbor in 0..num_points {
            if !is_visited[neighbor] && lp_solution[edge_vars[cur_point][neighbor]].round() == 1.0 {
                cur_point = neighbor;
                break;
            }
        }
    }
    assert_eq!(tour.len(), num_points);
    Tour(tour)
}

const USAGE: &str = "\
USAGE:
    tsp --help
    tsp [--svg-output] INPUT_FILE

INPUT_FILE is a problem description in TSPLIB format. You can download some
problems from http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/.
Use - for stdin.

By default, prints a single line containing 1-based node indices in the
optimal tour order to stdout. If --svg-output option is enabled, prints an
SVG document containing the optimal tour.

Set RUST_LOG environment variable (e.g. to info) to enable logging to stderr.
";

fn main() {
    env_logger::init();

    let args = std::env::args().collect::<Vec<_>>();
    if args.len() <= 1 {
        eprint!("{}", USAGE);
        std::process::exit(1);
    }

    if args[1] == "--help" {
        eprintln!("Finds the optimal solution for a traveling salesman problem.\n");
        eprint!("{}", USAGE);
        return;
    }

    let (enable_svg_output, filename) = if args.len() == 2 {
        (false, &args[1])
    } else if args.len() == 3 && args[1] == "--svg-output" {
        (true, &args[2])
    } else {
        eprintln!("Failed to parse arguments.\n");
        eprint!("{}", USAGE);
        std::process::exit(1);
    };

    let problem = if filename == "-" {
        Problem::parse(std::io::stdin().lock()).unwrap()
    } else {
        let file = std::fs::File::open(filename).unwrap();
        let input = io::BufReader::new(file);
        Problem::parse(input).unwrap()
    };

    let tour = solve(&problem);
    if enable_svg_output {
        print!("{}", tour.to_svg(&problem));
    } else {
        println!("{}", tour.to_string());
    }
}
