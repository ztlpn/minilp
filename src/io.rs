use crate::{ComparisonOp, LinearExpr, OptimizationDirection, Problem, Variable};
use std::{
    collections::{HashMap, HashSet},
    io,
};

pub struct MpsFile {
    pub problem_name: String,
    pub variables: HashMap<String, Variable>,
    pub problem: Problem,
}

pub fn parse_mps_file<R: io::BufRead>(
    input: &mut R,
    direction: OptimizationDirection,
) -> io::Result<MpsFile> {
    // Format descriptions:
    // Introduction: http://lpsolve.sourceforge.net/5.5/mps-format.htm
    // More in-depth: http://cgm.cs.mcgill.ca/~avis/courses/567/cplex/reffileformatscplex.pdf

    struct Lines<R: io::BufRead> {
        input: R,
        cur: String,
        idx: usize,
    }

    impl<R: io::BufRead> Lines<R> {
        fn to_next(&mut self) -> io::Result<()> {
            loop {
                self.idx += 1;
                self.cur.clear();
                self.input.read_line(&mut self.cur)?;
                if self.cur.is_empty() {
                    return Ok(());
                }

                if self.cur.starts_with("*") {
                    continue;
                }

                let len = self.cur.trim_end().len();
                if len != 0 {
                    self.cur.truncate(len);
                    return Ok(());
                }
            }
        }
    }

    let mut lines = Lines {
        input,
        cur: String::new(),
        idx: 0,
    };

    let problem_name = {
        lines.to_next()?;
        let mut tokens = lines.cur.split_whitespace();
        assert_eq!(tokens.next().unwrap(), "NAME");
        tokens.next().unwrap_or("").to_owned()
    };

    struct ConstraintDef {
        lhs: LinearExpr,
        cmp_op: ComparisonOp,
        rhs: f64,
        range: f64,
    }

    let mut cost_name = None;
    let mut free_rows = HashSet::new();
    let mut constraints = vec![];
    let mut constr_name2idx = HashMap::new();
    {
        lines.to_next()?;
        assert_eq!(lines.cur, "ROWS");

        loop {
            lines.to_next()?;
            if !lines.cur.starts_with(" ") {
                break;
            }

            let mut tokens = lines.cur.split_whitespace();
            let row_type = tokens.next().unwrap();
            let name = tokens.next().unwrap();
            let cmp_op = match row_type {
                "N" => {
                    if cost_name.is_none() {
                        cost_name = Some(name.to_owned());
                    } else {
                        free_rows.insert(name.to_owned());
                    }
                    continue;
                }
                "L" => ComparisonOp::Le,
                "G" => ComparisonOp::Ge,
                "E" => ComparisonOp::Eq,
                _ => panic!(),
            };
            assert!(constr_name2idx
                .insert(name.to_owned(), constraints.len())
                .is_none());
            constraints.push(ConstraintDef {
                lhs: LinearExpr::empty(),
                cmp_op,
                rhs: 0.0,
                range: 0.0,
            });
        }
    }

    let cost_name = cost_name.unwrap();

    #[derive(Default)]
    struct VariableDef {
        min: Option<f64>,
        max: Option<f64>,
        obj_coeff: Option<f64>,
    }

    let mut var_defs = vec![];
    let mut var_name2idx = HashMap::new();
    {
        assert_eq!(lines.cur, "COLUMNS");

        let mut cur_var = Variable(0);
        let mut cur_name = String::new();
        let mut cur_def = VariableDef::default();
        loop {
            lines.to_next()?;
            if !lines.cur.starts_with(" ") {
                break;
            }

            let mut tokens = lines.cur.split_whitespace();
            let name = tokens.next().unwrap();

            if name != cur_name {
                if !cur_name.is_empty() {
                    assert!(var_name2idx
                        .insert(std::mem::take(&mut cur_name), cur_var)
                        .is_none());
                    var_defs.push(std::mem::take(&mut cur_def));
                    cur_var.0 += 1;
                }
                cur_name = name.to_owned();
            }

            let coeff_tokens = tokens.collect::<Vec<_>>();
            for chunk in coeff_tokens.chunks(2) {
                assert_eq!(chunk.len(), 2);
                let coeff = chunk[1].parse::<f64>().unwrap();
                if chunk[0] == cost_name {
                    assert!(cur_def.obj_coeff.replace(coeff).is_none());
                } else if let Some(idx) = constr_name2idx.get(chunk[0]) {
                    constraints[*idx].lhs.add(cur_var, coeff);
                } else if free_rows.get(chunk[0]).is_none() {
                    panic!("unknown constraint: {}", chunk[0]);
                }
            }
        }

        if !cur_name.is_empty() {
            assert!(var_name2idx.insert(cur_name, cur_var).is_none());
            var_defs.push(std::mem::take(&mut cur_def));
        }
    }

    {
        assert_eq!(lines.cur, "RHS");

        let mut cur_vec_name = None;
        loop {
            lines.to_next()?;
            if !lines.cur.starts_with(" ") {
                break;
            }

            let mut tokens = lines.cur.split_whitespace();
            let vec_name = tokens.next().unwrap();

            if cur_vec_name.is_none() {
                cur_vec_name = Some(vec_name.to_owned());
            } else if cur_vec_name.as_deref() != Some(vec_name) {
                // use only the first RHS vector
                continue;
            }

            let rhs_tokens = tokens.collect::<Vec<_>>();
            for chunk in rhs_tokens.chunks(2) {
                assert_eq!(chunk.len(), 2);
                let rhs = chunk[1].parse::<f64>().unwrap();
                if chunk[0] == cost_name {
                    unimplemented!();
                } else if let Some(idx) = constr_name2idx.get(chunk[0]) {
                    constraints[*idx].rhs = rhs;
                } else {
                    panic!("unknown constraint: {}", chunk[0]);
                }
            }
        }
    }

    if lines.cur == "RANGES" {
        let mut cur_vec_name = None;
        loop {
            lines.to_next()?;
            if !lines.cur.starts_with(" ") {
                break;
            }

            let mut tokens = lines.cur.split_whitespace();

            let vec_name = tokens.next().unwrap();
            if cur_vec_name.is_none() {
                cur_vec_name = Some(vec_name.to_owned());
            } else if cur_vec_name.as_deref() != Some(vec_name) {
                // use only the first RANGES vector
                continue;
            }

            let rhs_tokens = tokens.collect::<Vec<_>>();
            for chunk in rhs_tokens.chunks(2) {
                assert_eq!(chunk.len(), 2);
                let range = chunk[1].parse::<f64>().unwrap();
                if let Some(idx) = constr_name2idx.get(chunk[0]) {
                    constraints[*idx].range = range;
                } else {
                    panic!("unknown constraint: {}", chunk[0]);
                }
            }
        }
    }

    if lines.cur == "BOUNDS" {
        let mut cur_vec_name = None;
        loop {
            lines.to_next()?;
            if !lines.cur.starts_with(" ") {
                break;
            }

            let mut tokens = lines.cur.split_whitespace();

            let bound_type = tokens.next().unwrap();
            if bound_type != "LO" && bound_type != "UP" && bound_type != "FX" {
                unimplemented!(
                    "line {}: bound type {} not supported",
                    lines.idx,
                    bound_type
                );
            }

            let vec_name = tokens.next().unwrap();
            if cur_vec_name.is_none() {
                cur_vec_name = Some(vec_name.to_owned());
            } else if cur_vec_name.as_deref() != Some(vec_name) {
                // use only the first BOUNDS vector
                continue;
            }

            let var_name = tokens.next().unwrap();
            let var_idx = var_name2idx.get(var_name).unwrap();
            let var_def = &mut var_defs[var_idx.0];
            let val = tokens.next().unwrap().parse::<f64>().unwrap();
            match bound_type {
                "LO" => assert!(var_def.min.replace(val).is_none()),
                "UP" => assert!(var_def.max.replace(val).is_none()),
                "FX" => {
                    assert!(var_def.min.replace(val).is_none());
                    assert!(var_def.max.replace(val).is_none());
                }
                _ => unreachable!(),
            }
        }
    }

    assert_eq!(lines.cur, "ENDATA");

    let mut problem = Problem::new(direction);
    for var_def in &var_defs {
        let (min, max) = match (var_def.min, var_def.max) {
            (Some(min), Some(max)) => (min, max),
            (Some(min), None) => (min, f64::INFINITY),
            (None, Some(max)) if max < 0.0 => (f64::NEG_INFINITY, max),
            (None, Some(max)) => (0.0, max),
            (None, None) => (0.0, f64::INFINITY),
        };
        problem.add_var((min, max), var_def.obj_coeff.unwrap_or(0.0));
    }
    for constr in constraints {
        if constr.range == 0.0 {
            problem.add_constraint(constr.lhs, constr.cmp_op, constr.rhs);
        } else {
            let (min, max) = match constr.cmp_op {
                ComparisonOp::Ge => (constr.rhs, constr.rhs + constr.range.abs()),
                ComparisonOp::Le => (constr.rhs - constr.range.abs(), constr.rhs),
                ComparisonOp::Eq if constr.range > 0.0 => (constr.rhs, constr.rhs + constr.range),
                ComparisonOp::Eq => (constr.rhs + constr.range, constr.rhs),
            };
            problem.add_constraint(constr.lhs.clone(), ComparisonOp::Ge, min);
            problem.add_constraint(constr.lhs, ComparisonOp::Le, max);
        }
    }

    Ok(MpsFile {
        problem_name,
        variables: var_name2idx,
        problem,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_FILE: &str = "\
* test file
NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
COLUMNS
    XONE      COST                 1   LIM1                 1
    XONE      LIM2                 1

    YTWO      COST                 4   LIM1                 1
    YTWO      MYEQN               -1

    ZTHREE    COST                 9   LIM2                 1
    ZTHREE    MYEQN                1
RHS
    RHS1      LIM1                 5   LIM2                10
    RHS1      MYEQN                7
BOUNDS
 UP BND1      XONE                 4
 LO BND1      YTWO                -1
 UP BND1      YTWO                 1
ENDATA
";

    #[test]
    fn test_parse_mps_file() {
        let mut input = io::Cursor::new(TEST_FILE);
        let file = parse_mps_file(&mut input, OptimizationDirection::Minimize).unwrap();
        assert_eq!(file.problem_name, "TESTPROB");
        assert_eq!(file.variables.len(), 3);

        let sol = file.problem.solve().unwrap();
        assert_eq!(sol[file.variables["XONE"]], 4.0);
        assert_eq!(sol[file.variables["YTWO"]], -1.0);
        assert_eq!(sol[file.variables["ZTHREE"]], 6.0);
        assert_eq!(sol.objective(), 54.0);
    }
}
