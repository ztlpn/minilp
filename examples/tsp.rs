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

    fn dist(&self, p1: usize, p2: usize) -> f64 {
        self.points[p1].dist(self.points[p2])
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

        let solution = lp_problem.solve().unwrap();
        eprintln!("objective: {}", solution.objective());
    }
}

fn main() {
    env_logger::init();

    let problem = Problem::read();
    problem.solve();
}
