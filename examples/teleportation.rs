//! This example sets up the classic quantum teleportation problem

use rand::{rngs::StdRng, SeedableRng};
use schrodinger::{
    gates::{cnot, h, x, z},
    state::StateVector,
};

fn main() {
    let input_bit = get_args();

    // prepare gates
    let x_gate = x();
    let h_gate = h();
    let z_gate = z();
    let cn_gate = cnot();

    // initialize simulator
    let mut sim = StateVector::zero(3);
    let mut rng = StdRng::seed_from_u64(0);

    // entangle alice and bob's states
    sim.apply_1qubit_gate_naive(&h_gate, 0);
    sim.apply_2qubit_gate_naive(&cn_gate, 0, 2);

    if input_bit == 1 {
        sim.apply_1qubit_gate_naive(&x_gate, 1); // set alice's qubit to input bit
    }

    sim.apply_2qubit_gate_naive(&cn_gate, 0, 1);
    sim.apply_1qubit_gate_naive(&h_gate, 0);

    let q = sim.measure_one_qubit(0, &mut rng) & 1; // get alice's classical measurements
    let a = sim.measure_one_qubit(1, &mut rng) & 1;

    // decode on bob's side
    if a == 1 {
        sim.apply_1qubit_gate_naive(&x_gate, 2);
    };
    if q == 1 {
        sim.apply_1qubit_gate_naive(&z_gate, 2);
    }

    // end of classical problem, bob now has alice's quantum state, but we measure
    let b = sim.measure_one_qubit(2, &mut rng) & 1;

    // show output
    println!("Before: Alice has {input_bit}. After: Bob has {b}.");
}

fn get_args() -> usize {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        eprintln!("usage: cargo run --release --example teleportation -- <alice's bit>");
        std::process::exit(1);
    }
    match args[0].as_str() {
        "0" => 0,
        "1" => 1,
        other => {
            eprintln!("expected 0 or 1, got {other:?}");
            std::process::exit(1);
        }
    }
}
