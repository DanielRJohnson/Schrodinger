use ndarray::Array2;
use num_complex::Complex64;
use shrodinger::{gates, state::StateVector};
use std::f64::consts::FRAC_1_SQRT_2;

const TOL: f64 = 1e-12;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn assert_state(s: &StateVector, expected: &[Complex64]) {
    let amps = s.amplitudes();
    assert_eq!(amps.len(), expected.len(), "wrong state length");
    for (i, (a, e)) in amps.iter().zip(expected).enumerate() {
        assert!((a - e).norm() < TOL, "amp[{i}]: got {a}, expected {e}");
    }
}

/// A generic-ish state: every amplitude nonzero with varied phases.
fn scrambled(n: usize) -> StateVector {
    let mut s = StateVector::zero(n);
    for q in 0..n {
        s.apply_1qubit_gate_naive(&gates::h(), q);
        s.apply_1qubit_gate_naive(&gates::t(), q);
        s.apply_1qubit_gate_naive(&gates::h(), q);
        s.apply_1qubit_gate_naive(&gates::s(), q);
    }
    s
}

fn all_gates() -> Vec<(&'static str, Array2<Complex64>)> {
    vec![
        ("x", gates::x()),
        ("y", gates::y()),
        ("z", gates::z()),
        ("h", gates::h()),
        ("s", gates::s()),
        ("t", gates::t()),
    ]
}

// ---------- the gate matrices themselves ----------

#[test]
fn every_gate_is_unitary() {
    for (name, g) in all_gates() {
        let g_dagger = g.t().mapv(|z| z.conj());
        let product = g_dagger.dot(&g);
        let id = Array2::<Complex64>::eye(2);
        for (a, b) in product.iter().zip(id.iter()) {
            assert!((a - b).norm() < TOL, "{name}: U†U != I");
        }
    }
}

// ---------- action on |0> ----------

#[test]
fn gates_on_zero_state() {
    let s2 = FRAC_1_SQRT_2;
    // (gate, expected amplitudes for [|0>, |1>])
    let cases: Vec<(&str, Array2<Complex64>, [Complex64; 2])> = vec![
        ("x", gates::x(), [c(0.0, 0.0), c(1.0, 0.0)]),
        ("y", gates::y(), [c(0.0, 0.0), c(0.0, 1.0)]), // Y|0> = i|1>
        ("z", gates::z(), [c(1.0, 0.0), c(0.0, 0.0)]),
        ("h", gates::h(), [c(s2, 0.0), c(s2, 0.0)]),
        ("s", gates::s(), [c(1.0, 0.0), c(0.0, 0.0)]),
        ("t", gates::t(), [c(1.0, 0.0), c(0.0, 0.0)]),
    ];
    for (name, g, expected) in cases {
        let mut s = StateVector::zero(1);
        s.apply_1qubit_gate_naive(&g, 0);
        let amps = s.amplitudes();
        for i in 0..2 {
            assert!(
                (amps[i] - expected[i]).norm() < TOL,
                "{name} on |0>, amp[{i}]"
            );
        }
    }
}

// ---------- action on |1> (prepared with X) ----------

#[test]
fn gates_on_one_state() {
    let s2 = FRAC_1_SQRT_2;
    let t_phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let cases: Vec<(&str, Array2<Complex64>, [Complex64; 2])> = vec![
        ("x", gates::x(), [c(1.0, 0.0), c(0.0, 0.0)]),
        ("y", gates::y(), [c(0.0, -1.0), c(0.0, 0.0)]), // Y|1> = -i|0>
        ("z", gates::z(), [c(0.0, 0.0), c(-1.0, 0.0)]), // Z|1> = -|1>
        ("h", gates::h(), [c(s2, 0.0), c(-s2, 0.0)]),
        ("s", gates::s(), [c(0.0, 0.0), c(0.0, 1.0)]), // S|1> = i|1>
        ("t", gates::t(), [c(0.0, 0.0), t_phase]),     // T|1> = e^{iπ/4}|1>
    ];
    for (name, g, expected) in cases {
        let mut s = StateVector::zero(1);
        s.apply_1qubit_gate_naive(&gates::x(), 0);
        s.apply_1qubit_gate_naive(&g, 0);
        let amps = s.amplitudes();
        for i in 0..2 {
            assert!(
                (amps[i] - expected[i]).norm() < TOL,
                "{name} on |1>, amp[{i}]"
            );
        }
    }
}

// ---------- algebraic identities ----------

#[test]
fn self_inverse_gates_square_to_identity() {
    for (name, g) in [
        ("x", gates::x()),
        ("y", gates::y()),
        ("z", gates::z()),
        ("h", gates::h()),
    ] {
        let mut s = scrambled(3);
        let before = s.amplitudes().clone();
        s.apply_1qubit_gate_naive(&g, 1);
        s.apply_1qubit_gate_naive(&g, 1);
        for (a, b) in s.amplitudes().iter().zip(before.iter()) {
            assert!((a - b).norm() < TOL, "{name}² != I");
        }
    }
}

#[test]
fn s_squared_is_z_and_t_squared_is_s() {
    let mut a = scrambled(2);
    let mut b = scrambled(2);
    a.apply_1qubit_gate_naive(&gates::s(), 0);
    a.apply_1qubit_gate_naive(&gates::s(), 0);
    b.apply_1qubit_gate_naive(&gates::z(), 0);
    for (x, y) in a.amplitudes().iter().zip(b.amplitudes().iter()) {
        assert!((x - y).norm() < TOL, "S² != Z");
    }

    let mut a = scrambled(2);
    let mut b = scrambled(2);
    a.apply_1qubit_gate_naive(&gates::t(), 1);
    a.apply_1qubit_gate_naive(&gates::t(), 1);
    b.apply_1qubit_gate_naive(&gates::s(), 1);
    for (x, y) in a.amplitudes().iter().zip(b.amplitudes().iter()) {
        assert!((x - y).norm() < TOL, "T² != S");
    }
}

#[test]
fn hadamard_conjugation_swaps_x_and_z() {
    // HZH = X
    let mut a = scrambled(2);
    let mut b = scrambled(2);
    for g in [gates::h(), gates::z(), gates::h()] {
        a.apply_1qubit_gate_naive(&g, 0);
    }
    b.apply_1qubit_gate_naive(&gates::x(), 0);
    for (x, y) in a.amplitudes().iter().zip(b.amplitudes().iter()) {
        assert!((x - y).norm() < TOL, "HZH != X");
    }

    // HXH = Z
    let mut a = scrambled(2);
    let mut b = scrambled(2);
    for g in [gates::h(), gates::x(), gates::h()] {
        a.apply_1qubit_gate_naive(&g, 0);
    }
    b.apply_1qubit_gate_naive(&gates::z(), 0);
    for (x, y) in a.amplitudes().iter().zip(b.amplitudes().iter()) {
        assert!((x - y).norm() < TOL, "HXH != Z");
    }
}

#[test]
fn xyz_equals_i_times_identity() {
    // Matrix product X·Y·Z = iI, so applying Z, then Y, then X multiplies the state by i.
    let mut s = scrambled(2);
    let before = s.amplitudes().clone();
    for g in [gates::z(), gates::y(), gates::x()] {
        s.apply_1qubit_gate_naive(&g, 0);
    }
    for (a, b) in s.amplitudes().iter().zip(before.iter()) {
        assert!((a - b * c(0.0, 1.0)).norm() < TOL, "XYZ != iI");
    }
}

#[test]
fn h_s_h_on_zero_matches_hand_computation() {
    // H|0> = (1,1)/√2 -> S -> (1,i)/√2 -> H -> ((1+i)/2, (1-i)/2)
    let mut s = StateVector::zero(1);
    for g in [gates::h(), gates::s(), gates::h()] {
        s.apply_1qubit_gate_naive(&g, 0);
    }
    assert_state(&s, &[c(0.5, 0.5), c(0.5, -0.5)]);
}

#[test]
fn gate_order_matters() {
    // Z then H: (1, -1)/√2.   H then Z: (1, 1)/√2 -> (1, -1)/√2... compare the other pair below.
    let s2 = FRAC_1_SQRT_2;

    let mut zh = StateVector::zero(1);
    zh.apply_1qubit_gate_naive(&gates::z(), 0);
    zh.apply_1qubit_gate_naive(&gates::h(), 0);
    assert_state(&zh, &[c(s2, 0.0), c(s2, 0.0)]); // Z|0> = |0>, then H

    let mut hz = StateVector::zero(1);
    hz.apply_1qubit_gate_naive(&gates::h(), 0);
    hz.apply_1qubit_gate_naive(&gates::z(), 0);
    assert_state(&hz, &[c(s2, 0.0), c(-s2, 0.0)]); // H|0>, then Z flips the sign of |1>
}

// ---------- targeting the right qubit in a multi-qubit register ----------

#[test]
fn x_flips_the_expected_bit_on_every_qubit() {
    let n = 4;
    for q in 0..n {
        let mut s = StateVector::zero(n);
        s.apply_1qubit_gate_naive(&gates::x(), q);
        let hot = 1usize << (n - 1 - q); // qubit 0 is the most significant bit
        for (i, a) in s.amplitudes().iter().enumerate() {
            let expected = if i == hot { 1.0 } else { 0.0 };
            assert!((a - c(expected, 0.0)).norm() < TOL, "q={q}, index {i}");
        }
    }
}

#[test]
fn h_on_one_qubit_only_touches_that_qubit() {
    let n = 3;
    for q in 0..n {
        let mut s = StateVector::zero(n);
        s.apply_1qubit_gate_naive(&gates::h(), q);
        let hot = 1usize << (n - 1 - q);
        for (i, a) in s.amplitudes().iter().enumerate() {
            let expected = if i == 0 || i == hot {
                FRAC_1_SQRT_2
            } else {
                0.0
            };
            assert!((a - c(expected, 0.0)).norm() < TOL, "q={q}, index {i}");
        }
    }
}

#[test]
fn h_on_every_qubit_gives_uniform_superposition() {
    let n = 3;
    let mut s = StateVector::zero(n);
    for q in 0..n {
        s.apply_1qubit_gate_naive(&gates::h(), q);
    }
    let amp = 1.0 / ((1usize << n) as f64).sqrt();
    for a in s.amplitudes().iter() {
        assert!((a - c(amp, 0.0)).norm() < TOL);
    }
}

#[test]
fn phase_gates_on_different_qubits_land_on_the_right_amplitudes() {
    // |++>, then S on qubit 0 and T on qubit 1.
    // Amplitude of |b0 b1> = (1/2) * i^{b0} * e^{iπ b1 / 4}.
    let mut s = StateVector::zero(2);
    for q in 0..2 {
        s.apply_1qubit_gate_naive(&gates::h(), q);
    }
    s.apply_1qubit_gate_naive(&gates::s(), 0);
    s.apply_1qubit_gate_naive(&gates::t(), 1);

    let t_phase = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
    let expected = [
        c(0.5, 0.0),           // |00>
        c(0.5, 0.0) * t_phase, // |01>
        c(0.0, 0.5),           // |10>
        c(0.0, 0.5) * t_phase, // |11>
    ];
    assert_state(&s, &expected);
}

// ---------- invariants ----------

#[test]
fn every_gate_on_every_qubit_preserves_norm() {
    let n = 4;
    for (name, g) in all_gates() {
        for q in 0..n {
            let mut s = scrambled(n);
            s.apply_1qubit_gate_naive(&g, q);
            let total: f64 = s.probabilities().iter().sum();
            assert!(
                (total - 1.0).abs() < TOL,
                "{name} on qubit {q}: norm = {total}"
            );
        }
    }
}
