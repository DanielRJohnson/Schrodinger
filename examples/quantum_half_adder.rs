//! This example sets up a quantum half-adder.
//! Circuit:
//! ```text
//! q0 - X o - o -
//! q1 - X - o o -
//! q2 - - + + - measure
//! q3 - - - - + measure
//! ```

use rand::{rngs::StdRng, SeedableRng};
use shrodinger::{
    gates::{cnot, toffoli, x},
    state::StateVector,
};

fn main() {
    let (a, b) = get_bit_args();

    // prepare gates
    let x_gate = x();
    let cn_gate = cnot();
    let tf_gate = toffoli();

    // initialize simulator
    let mut sim = StateVector::zero(4);

    // execute circuit
    if a == 1 {
        sim.apply_1qubit_gate_naive(&x_gate, 0); // set q0 = a
    }
    if b == 1 {
        sim.apply_1qubit_gate_naive(&x_gate, 1); // set q1 = b
    }
    sim.apply_2qubit_gate_naive(&cn_gate, 0, 2);
    sim.apply_2qubit_gate_naive(&cn_gate, 1, 2);
    sim.apply_3qubit_gate_naive(&tf_gate, 0, 1, 3);

    // measure output
    let mut rng = StdRng::seed_from_u64(0);
    let sum = sim.measure_one_qubit(2, &mut rng);
    let carry = sim.measure_one_qubit(3, &mut rng);

    // show output
    println!("{a} + {b} = {carry}{sum} (sum = {sum}, carry = {carry})");
}

fn get_bit_args() -> (usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: cargo run --example quantum_half_adder -- <a: 0|1> <b: 0|1>");
        std::process::exit(1);
    }
    let bit = |s: &String| -> usize {
        match s.as_str() {
            "0" => 0,
            "1" => 1,
            other => {
                eprintln!("expected 0 or 1, got {other:?}");
                std::process::exit(1);
            }
        }
    };
    (bit(&args[0]), bit(&args[1]))
}
