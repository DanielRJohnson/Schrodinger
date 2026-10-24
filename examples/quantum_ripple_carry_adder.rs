//! This example sets up a 4-bit quantum ripple-carry adder
//!
//! Qubit layout (12 qubits, bit 0 is the least significant):
//!   a register:     qubits 0..4     (unchanged)
//!   b register:     qubits 4..8     (overwritten with the sum bits)
//!   carry register: qubits 8..12    (carry out of bit i is qubit 8 + i)
//!
//! For each bit i, from least to most significant:
//! ```text
//! a_i   - o - o -------
//! b_i   - o + - o - + -    (b_i ends up holding the sum bit)
//! c_i   - - - - o - o -    (carry into bit i; skipped for i = 0, where it is 0)
//! c_i+1 - + - - + - - -    (carry out of bit i)
//! ```
//! with the gate order Toffoli(a_i, b_i -> c_i+1), CNOT(a_i -> b_i),
//! Toffoli(c_i, b_i -> c_i+1), CNOT(c_i -> b_i).
//!
//! The answer is read from the b register plus the final carry qubit.

use rand::{rngs::StdRng, SeedableRng};
use shrodinger::{
    gates::{cnot, toffoli, x},
    state::StateVector,
};

const BITS: usize = 4;
const N: usize = 3 * BITS;

fn main() {
    let (a, b) = get_args();

    // qubit layout
    let a_q = |i: usize| i;
    let b_q = |i: usize| BITS + i;
    let carry_q = |i: usize| 2 * BITS + i - 1; // carry INTO bit i, for i in 1..=BITS

    // prepare gates
    let x_gate = x();
    let cn_gate = cnot();
    let tf_gate = toffoli();

    // initialize simulator
    let mut sim = StateVector::zero(N);

    // load inputs
    for i in 0..BITS {
        if bit(a, i) == 1 {
            sim.apply_1qubit_gate_naive(&x_gate, a_q(i));
        }
        if bit(b, i) == 1 {
            sim.apply_1qubit_gate_naive(&x_gate, b_q(i));
        }
    }

    // ripple the carry from the least to the most significant bit
    for i in 0..BITS {
        let carry_out = carry_q(i + 1);
        sim.apply_3qubit_gate_naive(&tf_gate, a_q(i), b_q(i), carry_out);
        sim.apply_2qubit_gate_naive(&cn_gate, a_q(i), b_q(i));
        if i > 0 {
            // bit 0 has no carry-in (it is always 0), so these two gates would do nothing
            let carry_in = carry_q(i);
            sim.apply_3qubit_gate_naive(&tf_gate, carry_in, b_q(i), carry_out);
            sim.apply_2qubit_gate_naive(&cn_gate, carry_in, b_q(i));
        }
    }

    // measure output: sum bits from the b register, then the final carry
    let mut rng = StdRng::seed_from_u64(0);
    let mut result = 0;
    let mut weight = 1;
    for i in 0..BITS {
        result += weight * sim.measure_one_qubit(b_q(i), &mut rng);
        weight *= 2;
    }
    result += weight * sim.measure_one_qubit(carry_q(BITS), &mut rng);

    // show output
    println!("{a} + {b} = {result}  ({BITS}-bit adder, {N} qubits)");
    assert_eq!(
        result,
        a + b,
        "quantum result disagrees with ordinary addition"
    );
}

/// Bit `i` of `x`, where bit 0 is the least significant.
fn bit(x: usize, i: usize) -> usize {
    x / 2usize.pow(i as u32) % 2
}

fn get_args() -> (usize, usize) {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: cargo run --release --example ripple_carry_adder -- <a> <b>");
        std::process::exit(1);
    }
    let parse = |s: &String| -> usize {
        s.parse().unwrap_or_else(|_| {
            eprintln!("expected a non-negative integer, got {s:?}");
            std::process::exit(1);
        })
    };
    let (a, b) = (parse(&args[0]), parse(&args[1]));

    let limit = 2usize.pow(BITS as u32);
    if a >= limit || b >= limit {
        eprintln!("with {BITS} bits, a and b must be less than {limit}");
        std::process::exit(1);
    }
    (a, b)
}
