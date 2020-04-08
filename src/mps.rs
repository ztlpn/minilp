use crate::{ComparisonOp, LinearExpr, OptimizationDirection, Problem, Variable};
use std::{
    collections::{HashMap, HashSet},
    io,
};

/// A linear programming problem parsed from an MPS file.
#[derive(Clone)]
pub struct MpsFile {
    /// Value of the NAME field.
    pub problem_name: String,
    /// A mapping of a variable name to the corresponding [`Variable`].
    pub variables: HashMap<String, Variable>,
    /// A parsed problem.
    pub problem: Problem,
}

impl std::fmt::Debug for MpsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MpsFile")
            .field("problem_name", &self.problem_name)
            .field("problem", &self.problem)
            .finish()
    }
}

impl MpsFile {
    /// Parses a linear programming problem from an MPS file.
    ///
    /// This function supports the "free" MPS format, meaning that lines are tokenized based on
    /// whitespace, not based on position. Also, because MPS lacks any way to indicate
    /// the optimization direction, you have to supply it manually.
    ///
    /// # Errors
    ///
    /// Apart from I/O errors coming from `input`, this function will signal any syntax error
    /// as [`std::io::Error`] with the kind set to [`InvalidData`](std::io::ErrorKind::InvalidData).
    /// Unsupported features such as integer variables are reported similarly.
    pub fn parse<R: io::BufRead>(input: R, direction: OptimizationDirection) -> io::Result<Self> {
        // Format descriptions:
        // Introduction: http://lpsolve.sourceforge.net/5.5/mps-format.htm
        // More in-depth: http://cgm.cs.mcgill.ca/~avis/courses/567/cplex/reffileformatscplex.pdf

        let mut lines = Lines {
            input,
            cur: String::new(),
            idx: 0,
        };

        let problem_name = {
            lines.to_next()?;
            let mut tokens = Tokens::new(&lines);
            if tokens.next()? != "NAME" {
                return Err(lines.err("expected NAME section"));
            }
            tokens.iter.next().unwrap_or("").to_owned()
        };

        struct ConstraintDef {
            lhs: LinearExpr,
            cmp_op: ComparisonOp,
            rhs: f64,
            range: f64,
        }

        let mut obj_func_name = None;
        let mut free_rows = HashSet::new();
        let mut constraints = vec![];
        let mut constr_name2idx = HashMap::new();
        {
            lines.to_next()?;
            if lines.cur != "ROWS" {
                return Err(lines.err("expected ROWS section"));
            }

            loop {
                lines.to_next()?;
                if !lines.cur.starts_with(" ") {
                    break;
                }

                let mut tokens = Tokens::new(&lines);
                let row_type = tokens.next()?;
                let name = tokens.next()?;
                let cmp_op = match row_type {
                    "N" => {
                        if obj_func_name.is_none() {
                            obj_func_name = Some(name.to_owned());
                        } else {
                            free_rows.insert(name.to_owned());
                        }
                        continue;
                    }
                    "L" => ComparisonOp::Le,
                    "G" => ComparisonOp::Ge,
                    "E" => ComparisonOp::Eq,
                    _ => return Err(lines.err(&format!("unexpected row type {}", row_type))),
                };

                if constr_name2idx
                    .insert(name.to_owned(), constraints.len())
                    .is_some()
                {
                    return Err(lines.err(&format!("row {} already declared", name)));
                }

                constraints.push(ConstraintDef {
                    lhs: LinearExpr::empty(),
                    cmp_op,
                    rhs: 0.0,
                    range: 0.0,
                });
            }
        }

        let obj_func_name = if let Some(name) = obj_func_name {
            name
        } else {
            return Err(lines.err("objective function name not declared"));
        };

        #[derive(Default)]
        struct VariableDef {
            min: Option<f64>,
            max: Option<f64>,
            obj_coeff: f64,
        }

        let mut var_defs = vec![];
        let mut var_name2idx = HashMap::new();
        {
            if lines.cur != "COLUMNS" {
                return Err(lines.err("expected COLUMNS section"));
            }

            let mut cur_var = Variable(0);
            let mut cur_name = String::new();
            let mut cur_def = VariableDef::default();
            loop {
                lines.to_next()?;
                if !lines.cur.starts_with(" ") {
                    break;
                }

                let mut tokens = Tokens::new(&lines);
                let name = tokens.next()?;

                if name != cur_name {
                    if var_name2idx.get(name).is_some() {
                        return Err(lines.err(&format!("variable {} already declared", name)));
                    }

                    if !cur_name.is_empty() {
                        var_name2idx.insert(std::mem::take(&mut cur_name), cur_var);
                        var_defs.push(std::mem::take(&mut cur_def));
                        cur_var.0 += 1;
                    }
                    cur_name = name.to_owned();
                }

                for (key, val) in KVPairs::parse(&mut tokens)?.iter() {
                    if key == obj_func_name {
                        cur_def.obj_coeff = val;
                    } else if let Some(idx) = constr_name2idx.get(key) {
                        constraints[*idx].lhs.add(cur_var, val);
                    } else if free_rows.get(key).is_none() {
                        return Err(lines.err(&format!("unknown constraint: {}", key)));
                    }
                }
            }

            if !cur_name.is_empty() {
                var_name2idx.insert(std::mem::take(&mut cur_name), cur_var);
                var_defs.push(std::mem::take(&mut cur_def));
            }
        }

        {
            if lines.cur != "RHS" {
                return Err(lines.err("expected RHS section"));
            }

            let mut cur_vec_name = None;
            loop {
                lines.to_next()?;
                if !lines.cur.starts_with(" ") {
                    break;
                }

                let mut tokens = Tokens::new(&lines);
                let vec_name = tokens.next()?;

                if cur_vec_name.is_none() {
                    cur_vec_name = Some(vec_name.to_owned());
                } else if cur_vec_name.as_deref() != Some(vec_name) {
                    // use only the first RHS vector
                    continue;
                }

                for (key, val) in KVPairs::parse(&mut tokens)?.iter() {
                    if key == obj_func_name {
                        return Err(lines.err("setting objective in RHS section is not supported"));
                    } else if let Some(idx) = constr_name2idx.get(key) {
                        constraints[*idx].rhs = val;
                    } else {
                        return Err(lines.err(&format!("unknown constraint: {}", key)));
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

                let mut tokens = Tokens::new(&lines);

                let vec_name = tokens.next()?;
                if cur_vec_name.is_none() {
                    cur_vec_name = Some(vec_name.to_owned());
                } else if cur_vec_name.as_deref() != Some(vec_name) {
                    // use only the first RANGES vector
                    continue;
                }

                for (key, val) in KVPairs::parse(&mut tokens)?.iter() {
                    if let Some(idx) = constr_name2idx.get(key) {
                        constraints[*idx].range = val;
                    } else {
                        return Err(lines.err(&format!("unknown constraint: {}", key)));
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

                let mut tokens = Tokens::new(&lines);

                let bound_type = tokens.next()?;

                let vec_name = tokens.next()?;
                if cur_vec_name.is_none() {
                    cur_vec_name = Some(vec_name.to_owned());
                } else if cur_vec_name.as_deref() != Some(vec_name) {
                    // use only the first BOUNDS vector
                    continue;
                }

                let var_name = tokens.next()?;
                let var_idx = if let Some(idx) = var_name2idx.get(var_name) {
                    idx
                } else {
                    return Err(lines.err(&format!("unknown variable: {}", var_name)));
                };
                let var_def = &mut var_defs[var_idx.0];

                if bound_type == "FR" {
                    var_def.min = Some(f64::NEG_INFINITY);
                    var_def.max = Some(f64::INFINITY);
                    continue;
                } else {
                    let val = parse_f64(tokens.next()?, lines.idx)?;
                    match bound_type {
                        "LO" => var_def.min = Some(val),
                        "UP" => var_def.max = Some(val),
                        "FX" => {
                            var_def.min = Some(val);
                            var_def.max = Some(val);
                        }
                        _ => {
                            return Err(lines.err(&format!("bound type {} is not supported", bound_type)));
                        }
                    }
                }
            }
        }

        if lines.cur != "ENDATA" {
            return Err(lines.err("expected ENDATA section"));
        }

        let mut problem = Problem::new(direction);

        for var_def in &var_defs {
            let (min, max) = match (var_def.min, var_def.max) {
                (Some(min), Some(max)) => (min, max),
                (Some(min), None) => (min, f64::INFINITY),
                (None, Some(max)) if max < 0.0 => (f64::NEG_INFINITY, max),
                (None, Some(max)) => (0.0, max),
                (None, None) => (0.0, f64::INFINITY),
            };
            problem.add_var(var_def.obj_coeff, (min, max));
        }

        for constr in constraints {
            if constr.range == 0.0 {
                problem.add_constraint(constr.lhs, constr.cmp_op, constr.rhs);
            } else {
                let (min, max) = match constr.cmp_op {
                    ComparisonOp::Ge => (constr.rhs, constr.rhs + constr.range.abs()),
                    ComparisonOp::Le => (constr.rhs - constr.range.abs(), constr.rhs),
                    ComparisonOp::Eq if constr.range > 0.0 => {
                        (constr.rhs, constr.rhs + constr.range)
                    }
                    ComparisonOp::Eq => (constr.rhs + constr.range, constr.rhs),
                };
                problem.add_constraint(constr.lhs.clone(), ComparisonOp::Ge, min);
                problem.add_constraint(constr.lhs, ComparisonOp::Le, max);
            }
        }

        Ok(Self {
            problem_name,
            variables: var_name2idx,
            problem,
        })
    }
}

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

    fn err(&self, msg: &str) -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("line {}: {}", self.idx, msg),
        )
    }
}

struct Tokens<'a> {
    line_idx: usize,
    iter: std::str::SplitWhitespace<'a>,
}

impl<'a> Tokens<'a> {
    fn new<R: io::BufRead>(lines: &'a Lines<R>) -> Self {
        Self {
            line_idx: lines.idx,
            iter: lines.cur.split_whitespace(),
        }
    }

    fn next(&mut self) -> io::Result<&'a str> {
        self.iter.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: unexpected end of line", self.line_idx),
            )
        })
    }
}

fn parse_f64(input: &str, line_idx: usize) -> io::Result<f64> {
    input.parse().or_else(|_| {
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "line {}: couldn't parse float from string: `{}`",
                line_idx, input
            ),
        ))
    })
}

struct KVPairs<'a> {
    // MPS allows one or two key-value pairs per line.
    first: (&'a str, f64),
    second: Option<(&'a str, f64)>,
}

impl<'a> KVPairs<'a> {
    fn parse(tokens: &mut Tokens<'a>) -> io::Result<Self> {
        let first_key = tokens.next()?;
        let first_val = parse_f64(tokens.next()?, tokens.line_idx)?;

        let second_key = if let Some(key) = tokens.iter.next() {
            key
        } else {
            return Ok(KVPairs {
                first: (first_key, first_val),
                second: None,
            });
        };
        let second_val = parse_f64(tokens.next()?, tokens.line_idx)?;
        Ok(KVPairs {
            first: (first_key, first_val),
            second: Some((second_key, second_val)),
        })
    }

    fn iter(self) -> impl Iterator<Item = (&'a str, f64)> {
        std::iter::once(self.first).chain(self.second)
    }
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
        let file = MpsFile::parse(&mut input, OptimizationDirection::Minimize).unwrap();
        assert_eq!(file.problem_name, "TESTPROB");
        assert_eq!(file.variables.len(), 3);

        let sol = file.problem.solve().unwrap();
        assert_eq!(sol[file.variables["XONE"]], 4.0);
        assert_eq!(sol[file.variables["YTWO"]], -1.0);
        assert_eq!(sol[file.variables["ZTHREE"]], 6.0);
        assert_eq!(sol.objective(), 54.0);
    }
}
