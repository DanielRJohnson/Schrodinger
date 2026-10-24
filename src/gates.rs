use std::f64::consts::FRAC_1_SQRT_2;

use ndarray::{array, Array2};
use num_complex::Complex64;

pub fn x() -> Array2<Complex64> {
    // X = [[0 1] [1 0]]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    array![[zero, one], [one, zero]]
}

pub fn y() -> Array2<Complex64> {
    // Y = [[0 -i] [i 0]]
    let zero = Complex64::new(0.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    array![[zero, -i], [i, zero]]
}

pub fn z() -> Array2<Complex64> {
    // Z = [[1 0] [0 -1]]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    array![[one, zero], [zero, -one]]
}

pub fn h() -> Array2<Complex64> {
    // H = 1/sqrt(2) .* [[1 1] [1 -1]]
    let h_elem = Complex64::new(FRAC_1_SQRT_2, 0.0);
    array![[h_elem, h_elem], [h_elem, -h_elem]]
}

pub fn s() -> Array2<Complex64> {
    // S = [[1 0] [0 i]]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    array![[one, zero], [zero, i]]
}

pub fn t() -> Array2<Complex64> {
    // T = [[1 0] [0 e^(i * pi / 4)]]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let t_exp = Complex64::new(0.0, 1.0) * Complex64::new(std::f64::consts::PI, 0.0)
        / Complex64::new(4.0, 0.0);
    let t_term = Complex64::new(std::f64::consts::E, 0.0).powc(t_exp);
    array![[one, zero], [zero, t_term]]
}

pub fn cnot() -> Array2<Complex64> {
    // CNOT = [ [1 0 0 0] [0 1 0 0] [0 0 0 1] [0 0 1 0] ]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    array![
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one],
        [zero, zero, one, zero]
    ]
}

pub fn cz() -> Array2<Complex64> {
    // CZ = [ [1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 -1] ]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    array![
        [one, zero, zero, zero],
        [zero, one, zero, zero],
        [zero, zero, one, zero],
        [zero, zero, zero, -one]
    ]
}

pub fn swap() -> Array2<Complex64> {
    // SWAP = [ [1 0 0 0] [0 0 1 0] [0 1 0 0] [0 0 0 1] ]
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    array![
        [one, zero, zero, zero],
        [zero, zero, one, zero],
        [zero, one, zero, zero],
        [zero, zero, zero, one]
    ]
}

pub fn toffoli() -> Array2<Complex64> {
    // TOFFOLI = I8x8, but you swap 7th and 8th rows
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let mut toff = Array2::<Complex64>::eye(8);
    toff[[6, 6]] = zero;
    toff[[7, 7]] = zero;
    toff[[6, 7]] = one;
    toff[[7, 6]] = one;
    toff
}
