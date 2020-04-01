# `tsp`

[Traveling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem) or TSP
is perhaps the most famous combinatorial optimization problem. Amazingly enough, even if linear
programming is a continuous optimization technique, with some ingenuity it can be used to solve
discrete problems such as TSP which makes it a great showcase for the `minilp` crate.

[`tsp.rs`](./tsp.rs) is a pretty basic solver, but it can still quickly produce *optimal* solutions
for problems up to 100-150 nodes in size. It accepts symmetric euclidean problems (meaning that the
distance the salesman must travel to get from one city to another is equal to the straight-line
distance between cities). Included is a sample problem [`bn130.tsp`](./bn130.tsp) (130 random
[blue noise](https://crates.io/crates/poisson) points) or you can try solving standard benchmark
problems from the [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/).

If you run
```
cargo run --release --example tsp -- --svg-output bn130.tsp > bn130.tsp.svg
```
you will get the following image depicting the optimal tour:
![optimal tour](./bn130.tsp.svg)