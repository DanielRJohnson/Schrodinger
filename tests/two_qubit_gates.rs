use ndarray::Array2;
use num_complex::Complex64;
use schrodinger::{gates, state::StateVector};

const TOL: f64 = 1e-12;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// Basis state |index> on n qubits, built with X gates (qubit 0 is the most significant bit).
fn basis(n: usize, index: usize) -> StateVector {
    let mut s = StateVector::zero(n);
    for q in 0..n {
        if (index >> (n - 1 - q)) & 1 == 1 {
            s.apply_1qubit_gate_naive(&gates::x(), q);
        }
    }
    s
}

/// Every amplitude nonzero, with varied phases.
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

fn assert_same(a: &StateVector, b: &StateVector, ctx: &str) {
    for (i, (x, y)) in a.amplitudes().iter().zip(b.amplitudes().iter()).enumerate() {
        assert!(
            (x - y).norm() < TOL,
            "{ctx}: index {i}: got {x}, expected {y}"
        );
    }
}

/// All amplitude on `index` with the given value, zero elsewhere.
fn assert_single(s: &StateVector, index: usize, value: Complex64, ctx: &str) {
    for (i, a) in s.amplitudes().iter().enumerate() {
        let expected = if i == index { value } else { c(0.0, 0.0) };
        assert!(
            (a - expected).norm() < TOL,
            "{ctx}: index {i}: got {a}, expected {expected}"
        );
    }
}

/// Run `gate` on every ordered pair (a, b) of a 3-qubit register and every basis input.
/// `expected(input, a_mask, b_mask)` returns the (output index, sign) for that input.
fn check_all_placements(
    name: &str,
    gate: &Array2<Complex64>,
    expected: impl Fn(usize, usize, usize) -> (usize, f64),
) {
    let n = 3;
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue;
            }
            let (am, bm) = (1 << (n - 1 - a), 1 << (n - 1 - b));
            for input in 0..(1 << n) {
                let mut s = basis(n, input);
                s.apply_2qubit_gate_naive(gate, a, b);
                let (out, sign) = expected(input, am, bm);
                assert_single(
                    &s,
                    out,
                    c(sign, 0.0),
                    &format!("{name}({a},{b}) on |{input:03b}>"),
                );
            }
        }
    }
}

// ---------- the matrices ----------

#[test]
fn all_gates_are_unitary() {
    for (name, g) in [
        ("cnot", gates::cnot()),
        ("cz", gates::cz()),
        ("swap", gates::swap()),
    ] {
        let product = g.t().mapv(|z| z.conj()).dot(&g);
        for (x, y) in product.iter().zip(Array2::<Complex64>::eye(4).iter()) {
            assert!((x - y).norm() < TOL, "{name}: U†U != I");
        }
    }
}

// ---------- truth tables on every ordered pair of qubits ----------

#[test]
fn cnot_flips_b_when_a_is_set() {
    check_all_placements("cnot", &gates::cnot(), |idx, am, bm| {
        (if idx & am != 0 { idx ^ bm } else { idx }, 1.0)
    });
}

#[test]
fn cz_negates_only_when_both_are_set() {
    check_all_placements("cz", &gates::cz(), |idx, am, bm| {
        (
            idx,
            if idx & am != 0 && idx & bm != 0 {
                -1.0
            } else {
                1.0
            },
        )
    });
}

#[test]
fn swap_exchanges_the_two_bits() {
    check_all_placements("swap", &gates::swap(), |idx, am, bm| {
        let differ = (idx & am != 0) != (idx & bm != 0);
        (if differ { idx ^ am ^ bm } else { idx }, 1.0)
    });
}

// ---------- algebraic properties, on a state with full support ----------

#[test]
fn every_gate_is_its_own_inverse() {
    for (name, g) in [
        ("cnot", gates::cnot()),
        ("cz", gates::cz()),
        ("swap", gates::swap()),
    ] {
        let mut s = scrambled(3);
        s.apply_2qubit_gate_naive(&g, 0, 2);
        s.apply_2qubit_gate_naive(&g, 0, 2);
        assert_same(&s, &scrambled(3), &format!("{name} applied twice"));
    }
}

#[test]
fn cz_and_swap_are_symmetric_in_their_qubits() {
    for (name, g) in [("cz", gates::cz()), ("swap", gates::swap())] {
        let (mut ab, mut ba) = (scrambled(3), scrambled(3));
        ab.apply_2qubit_gate_naive(&g, 0, 2);
        ba.apply_2qubit_gate_naive(&g, 2, 0);
        assert_same(&ab, &ba, &format!("{name}(0,2) vs {name}(2,0)"));
    }
}

#[test]
fn cz_equals_cnot_conjugated_by_h_on_the_target() {
    let (mut direct, mut built) = (scrambled(3), scrambled(3));
    direct.apply_2qubit_gate_naive(&gates::cz(), 0, 2);

    built.apply_1qubit_gate_naive(&gates::h(), 2);
    built.apply_2qubit_gate_naive(&gates::cnot(), 0, 2);
    built.apply_1qubit_gate_naive(&gates::h(), 2);
    assert_same(&built, &direct, "H·CNOT·H vs CZ");
}

#[test]
fn swap_equals_three_cnots() {
    let (mut direct, mut built) = (scrambled(3), scrambled(3));
    direct.apply_2qubit_gate_naive(&gates::swap(), 0, 2);

    built.apply_2qubit_gate_naive(&gates::cnot(), 0, 2);
    built.apply_2qubit_gate_naive(&gates::cnot(), 2, 0);
    built.apply_2qubit_gate_naive(&gates::cnot(), 0, 2);
    assert_same(&built, &direct, "three CNOTs vs SWAP");
}

// ---------- superposition inputs ----------

#[test]
fn h_then_cnot_makes_a_bell_state() {
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut s = StateVector::zero(2);
    s.apply_1qubit_gate_naive(&gates::h(), 0);
    s.apply_2qubit_gate_naive(&gates::cnot(), 0, 1);

    let expected = [c(s2, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(s2, 0.0)];
    for (i, (a, e)) in s.amplitudes().iter().zip(expected.iter()).enumerate() {
        assert!((a - e).norm() < TOL, "index {i}: got {a}, expected {e}");
    }
}

#[test]
fn cz_on_uniform_superposition_flips_only_the_last_sign() {
    let mut s = StateVector::zero(2);
    s.apply_1qubit_gate_naive(&gates::h(), 0);
    s.apply_1qubit_gate_naive(&gates::h(), 1);
    s.apply_2qubit_gate_naive(&gates::cz(), 0, 1);

    let expected = [c(0.5, 0.0), c(0.5, 0.0), c(0.5, 0.0), c(-0.5, 0.0)];
    for (i, (a, e)) in s.amplitudes().iter().zip(expected.iter()).enumerate() {
        assert!((a - e).norm() < TOL, "index {i}: got {a}, expected {e}");
    }
}
