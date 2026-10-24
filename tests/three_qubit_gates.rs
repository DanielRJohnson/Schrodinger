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

fn assert_single(s: &StateVector, index: usize, ctx: &str) {
    for (i, a) in s.amplitudes().iter().enumerate() {
        let expected = if i == index { c(1.0, 0.0) } else { c(0.0, 0.0) };
        assert!(
            (a - expected).norm() < TOL,
            "{ctx}: index {i}: got {a}, expected {expected}"
        );
    }
}

// ---------- the matrix ----------

#[test]
fn toffoli_matrix_is_unitary_and_self_inverse() {
    let g = gates::toffoli();
    let id = Array2::<Complex64>::eye(8);

    let dagger_product = g.t().mapv(|z| z.conj()).dot(&g);
    let square = g.dot(&g);
    for ((u, s), e) in dagger_product.iter().zip(square.iter()).zip(id.iter()) {
        assert!((u - e).norm() < TOL, "U†U != I");
        assert!((s - e).norm() < TOL, "U² != I");
    }
}

// ---------- truth table on every ordered triple of qubits ----------

#[test]
fn flips_target_only_when_both_controls_are_set() {
    let n = 4;
    for a in 0..n {
        for b in 0..n {
            for t in 0..n {
                if a == b || a == t || b == t {
                    continue;
                }
                let (am, bm, tm) = (1 << (n - 1 - a), 1 << (n - 1 - b), 1 << (n - 1 - t));
                for input in 0..(1usize << n) {
                    let mut s = basis(n, input);
                    s.apply_3qubit_gate_naive(&gates::toffoli(), a, b, t);

                    let output = if input & am != 0 && input & bm != 0 {
                        input ^ tm
                    } else {
                        input
                    };
                    assert_single(
                        &s,
                        output,
                        &format!("toffoli({a},{b},{t}) on |{input:04b}>"),
                    );
                }
            }
        }
    }
}

// ---------- algebraic properties, on a state with full support ----------

#[test]
fn applying_twice_restores_the_state() {
    let mut s = scrambled(4);
    s.apply_3qubit_gate_naive(&gates::toffoli(), 3, 0, 2);
    s.apply_3qubit_gate_naive(&gates::toffoli(), 3, 0, 2);
    assert_same(&s, &scrambled(4), "toffoli applied twice");
}

#[test]
fn the_two_controls_are_interchangeable() {
    let (mut ab, mut ba) = (scrambled(4), scrambled(4));
    ab.apply_3qubit_gate_naive(&gates::toffoli(), 0, 3, 1);
    ba.apply_3qubit_gate_naive(&gates::toffoli(), 3, 0, 1);
    assert_same(&ab, &ba, "toffoli(0,3,1) vs toffoli(3,0,1)");
}

#[test]
fn toffoli_equals_ccz_conjugated_by_h_on_the_target() {
    let mut ccz = Array2::<Complex64>::eye(8);
    ccz[[7, 7]] = c(-1.0, 0.0);

    let (mut direct, mut built) = (scrambled(4), scrambled(4));
    direct.apply_3qubit_gate_naive(&gates::toffoli(), 1, 3, 0);

    built.apply_1qubit_gate_naive(&gates::h(), 0);
    built.apply_3qubit_gate_naive(&ccz, 1, 3, 0);
    built.apply_1qubit_gate_naive(&gates::h(), 0);
    assert_same(&built, &direct, "H·CCZ·H vs Toffoli");
}

// ---------- superposition input ----------

#[test]
fn controls_in_superposition_flip_target_only_on_the_both_set_branch() {
    // H on qubits 0 and 1 gives (|000> + |010> + |100> + |110>) / 2.
    // Toffoli (controls 0, 1; target 2) moves |110> to |111>.
    let mut s = StateVector::zero(3);
    s.apply_1qubit_gate_naive(&gates::h(), 0);
    s.apply_1qubit_gate_naive(&gates::h(), 1);
    s.apply_3qubit_gate_naive(&gates::toffoli(), 0, 1, 2);

    let mut expected = [c(0.0, 0.0); 8];
    for idx in [0b000, 0b010, 0b100, 0b111] {
        expected[idx] = c(0.5, 0.0);
    }
    for (i, (a, e)) in s.amplitudes().iter().zip(expected.iter()).enumerate() {
        assert!((a - e).norm() < TOL, "index {i}: got {a}, expected {e}");
    }
}
