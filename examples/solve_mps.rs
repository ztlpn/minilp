use minilp::MpsFile;
use std::io;

const USAGE: &str = "\
Read a problem in the MPS format and solve it.

USAGE:
    solve_mps --help
    solve_mps INPUT_FILE

INPUT_FILE is a file in the M format. You can download some sample
problems from http://www.netlib.org/lp/data/. Use - for stdin.

Output is a single line containing the minimal objective value.

Set RUST_LOG environment variable (e.g. to info) to enable logging to stderr.
";

fn main() {
    env_logger::init();

    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        print!("{}", USAGE);
        std::process::exit(1);
    } else if args[1] == "--help" {
        print!("{}", USAGE);
        return;
    }

    let filename = &args[1];
    let direction = minilp::OptimizationDirection::Minimize;
    let file = if filename == "-" {
        MpsFile::parse(std::io::stdin().lock(), direction).unwrap()
    } else {
        let file = std::fs::File::open(filename).unwrap();
        let input = io::BufReader::new(file);
        MpsFile::parse(input, direction).unwrap()
    };

    let solution = file.problem.solve().unwrap();
    println!("objective value: {}", solution.objective());
}
