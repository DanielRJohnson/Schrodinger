use ndarray::linalg::kron;
use ndarray::{s, Array1, Array2};
use num_complex::Complex64;
use rand::{Rng, RngExt};

pub struct StateVector {
    n: usize,
    amplitudes: Array1<Complex64>,
}

impl StateVector {
    pub fn zero(n: usize) -> Self {
        let mut amplitudes = Array1::<Complex64>::zeros(1 << n);
        amplitudes[0] = Complex64::new(1.0, 0.0);
        Self { n, amplitudes }
    }

    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    pub fn amplitudes(&self) -> &Array1<Complex64> {
        &self.amplitudes
    }

    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        // get one state from the distribution of states
        let uniform: f64 = rng.random();
        let mut total = 0.0;
        for (i, a) in self.amplitudes.iter().enumerate() {
            total += a.norm_sqr(); // a^2 = prob
            if uniform < total {
                return i; // hit total, this is the state we return
            }
        }
        self.amplitudes.len() - 1 // reached the end
    }

    pub fn measure_all_qubits(&mut self, rng: &mut impl Rng) -> usize {
        // get a sample state and collapse the distribution to just that state
        let state = self.sample(rng);
        self.amplitudes.fill(Complex64::new(0.0, 0.0));
        self.amplitudes[state] = Complex64::new(1.0, 0.0);
        state
    }

    pub fn measure_one_qubit(&mut self, q: usize, rng: &mut impl Rng) -> usize {
        let mask = 1usize << (self.n - 1 - q); // bitstring mask of q's position

        let p1: f64 = self
            .amplitudes
            .iter()
            .enumerate()
            .filter(|(i, _)| i & mask != 0) // only keep states where q is 1
            .map(|(_, a)| a.norm_sqr())
            .sum(); // accumulate probability of q being 1

        let state = if rng.random::<f64>() < p1 { 1 } else { 0 }; // sample q based on prob

        // Collapse: keep only amplitudes whose bit q equals the outcome
        for (i, a) in self.amplitudes.iter_mut().enumerate() {
            if ((i & mask != 0) as usize) != state {
                *a = Complex64::new(0.0, 0.0);
            }
        }

        // Renormalize so distribution still sums to 1
        let norm = self
            .amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>()
            .sqrt();
        self.amplitudes.mapv_inplace(|a| a / norm);

        state // return q's sampled value
    }

    pub fn apply_1qubit_gate_naive(&mut self, gate: &Array2<Complex64>, q: usize) {
        // next state = (I <kprod> ... <kprod> U <kprod> ... <kprod> I) * state
        self.amplitudes = kron_chain(self.n, &[(q, gate)]).dot(&self.amplitudes);
    }

    pub fn apply_2qubit_gate_naive(&mut self, gate: &Array2<Complex64>, a: usize, b: usize) {
        // index is 2 * (bit of A) + (bit of B)
        // next state = (U + W + X + Y) * state
        // U = I <kprod> ... <kprod> E00 <kprod> ... <kprod> G00 <kprod> ... I
        // W = I <kprod> ... <kprod> E01 <kprod> ... <kprod> G01 <kprod> ... I
        // X = I <kprod> ... <kprod> E10 <kprod> ... <kprod> G10 <kprod> ... I
        // Y = I <kprod> ... <kprod> E11 <kprod> ... <kprod> G11 <kprod> ... I
        let mut m = Array2::<Complex64>::zeros((1 << self.n, 1 << self.n));
        for i in 0..2 {
            for j in 0..2 {
                let g_block = gate
                    .slice(s![2 * i..2 * i + 2, 2 * j..2 * j + 2]) // get the "corner" of gate
                    .to_owned();
                let mut selector = Array2::<Complex64>::zeros((2, 2));
                selector[[i, j]] = Complex64::new(1.0, 0.0);
                let chain = kron_chain(self.n, &[(a, &selector), (b, &g_block)]);
                m += &chain;
            }
        }
        self.amplitudes = m.dot(&self.amplitudes);
    }

    pub fn apply_3qubit_gate_naive(
        &mut self,
        gate: &Array2<Complex64>,
        a: usize,
        b: usize,
        c: usize,
    ) {
        // next state follows natually extending the bitstrings from 2-qubit application
        let mut m = Array2::<Complex64>::zeros((1 << self.n, 1 << self.n));
        for i in 0..4 {
            for j in 0..4 {
                let g_block = gate
                    .slice(s![2 * i..2 * i + 2, 2 * j..2 * j + 2]) // get the 2x2 subgate
                    .to_owned();
                let mut sel1 = Array2::<Complex64>::zeros((2, 2));
                sel1[[i >> 1, j >> 1]] = Complex64::new(1.0, 0.0); // high-bit = qubit a
                let mut sel2 = Array2::<Complex64>::zeros((2, 2));
                sel2[[i & 1, j & 1]] = Complex64::new(1.0, 0.0); // low-bit = qubit b
                let chain = kron_chain(self.n, &[(a, &sel1), (b, &sel2), (c, &g_block)]);
                m += &chain;
            }
        }
        self.amplitudes = m.dot(&self.amplitudes);
    }
}

fn kron_chain(n: usize, pos_mat_pairs: &[(usize, &Array2<Complex64>)]) -> Array2<Complex64> {
    // create a kron product chain with I at all values not specified in pos_mat_pairs
    // at the positions specified, put the specified matrices
    let mut prod = Array2::<Complex64>::eye(1);
    let id = Array2::eye(2);
    for i in 0..n {
        let mut pos_found = false;
        for (pos, mat) in pos_mat_pairs {
            if *pos == i {
                prod = kron(&prod, *mat);
                pos_found = true;
                break;
            }
        }
        if !pos_found {
            prod = kron(&prod, &id);
        }
    }
    prod
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{rngs::StdRng, SeedableRng};
    use std::f64::consts::FRAC_1_SQRT_2;

    #[test]
    fn zero_state_is_normalized() {
        let s = StateVector::zero(3);
        let total: f64 = s.probabilities().iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn probabilities_len_is_2_to_the_n() {
        let s1 = StateVector::zero(1);
        assert!(s1.probabilities().len() == 2);
        let s2 = StateVector::zero(2);
        assert!(s2.probabilities().len() == 4);
        let s3 = StateVector::zero(3);
        assert!(s3.probabilities().len() == 8);
        let s4 = StateVector::zero(4);
        assert!(s4.probabilities().len() == 16);
    }

    #[test]
    fn sample_returns_the_only_populated_index() {
        let mut amps = Array1::<Complex64>::zeros(4);
        amps[2] = Complex64::new(1.0, 0.0); // |10>
        let s = StateVector {
            n: 2,
            amplitudes: amps,
        };

        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(s.sample(&mut rng), 2);
    }

    #[test]
    fn measuring_a_bell_state_gives_matching_bits() {
        let h = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let zero = Complex64::new(0.0, 0.0);

        for seed in 0..10 {
            let mut s = StateVector {
                n: 2,
                amplitudes: Array1::from(vec![h, zero, zero, h]),
            };
            let mut rng = StdRng::seed_from_u64(seed);

            let first = s.measure_one_qubit(0, &mut rng);
            let second = s.measure_one_qubit(1, &mut rng);
            assert_eq!(first, second, "seed {seed}");
        }
    }

    #[test]
    fn measure_all_collapses_to_the_returned_basis_state() {
        let half = Complex64::new(0.5, 0.0);
        let mut s = StateVector {
            n: 2,
            amplitudes: Array1::from(vec![half; 4]),
        };

        let mut rng = StdRng::seed_from_u64(0);
        let outcome = s.measure_all_qubits(&mut rng);
        assert_eq!(s.probabilities()[outcome], 1.0);
    }

    fn x_gate() -> Array2<Complex64> {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        array![[zero, one], [one, zero]]
    }

    #[test]
    fn x_flips_bits() {
        // |00> -> |10>
        let mut s = StateVector::zero(2);
        s.apply_1qubit_gate_naive(&x_gate(), 0);
        assert_eq!(s.probabilities(), vec![0.0, 0.0, 1.0, 0.0]);

        // |00> -> |01>
        let mut s = StateVector::zero(2);
        s.apply_1qubit_gate_naive(&x_gate(), 1);
        assert_eq!(s.probabilities(), vec![0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn x_twice_is_identity() {
        // |000> -> |100> -> |000>
        let mut s = StateVector::zero(3);
        s.apply_1qubit_gate_naive(&x_gate(), 1);
        s.apply_1qubit_gate_naive(&x_gate(), 1);
        assert!((s.probabilities()[0] - 1.0).abs() < 1e-12);
    }

    fn cnot() -> Array2<Complex64> {
        let (z, o) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
        Array2::from_shape_vec((4, 4), vec![o, z, z, z, z, o, z, z, z, z, z, o, z, z, o, z])
            .unwrap()
    }

    #[test]
    fn truth_table() {
        // |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
        let expected_index = [0, 1, 3, 2];
        for (input, &output) in expected_index.iter().enumerate() {
            let mut amps = Array1::<Complex64>::zeros(4);
            amps[input] = Complex64::new(1.0, 0.0);

            let mut s = StateVector {
                n: 2,
                amplitudes: amps,
            };
            s.apply_2qubit_gate_naive(&cnot(), 0, 1);

            for (i, a) in s.amplitudes.iter().enumerate() {
                let expected = if i == output { 1.0 } else { 0.0 };
                assert!(
                    (a - Complex64::new(expected, 0.0)).norm() < 1e-12,
                    "input {input}, index {i}: got {a}"
                );
            }
        }
    }

    #[test]
    fn toffoli_truth_table_on_scattered_qubits() {
        let mut toffoli = Array2::<Complex64>::eye(8);
        toffoli[[6, 6]] = Complex64::new(0.0, 0.0);
        toffoli[[7, 7]] = Complex64::new(0.0, 0.0);
        toffoli[[6, 7]] = Complex64::new(1.0, 0.0);
        toffoli[[7, 6]] = Complex64::new(1.0, 0.0);

        let n = 4;
        let (a, b, c) = (3, 0, 2); // controls on qubits 3 and 0, target on qubit 2
        let (am, bm, cm) = (
            1usize << (n - 1 - a),
            1usize << (n - 1 - b),
            1usize << (n - 1 - c),
        );

        for input in 0..(1usize << n) {
            let mut amps = Array1::<Complex64>::zeros(1 << n);
            amps[input] = Complex64::new(1.0, 0.0);
            let mut s = StateVector {
                n,
                amplitudes: amps,
            };
            s.apply_3qubit_gate_naive(&toffoli, a, b, c);

            let output = if input & am != 0 && input & bm != 0 {
                input ^ cm
            } else {
                input
            };
            for (i, x) in s.amplitudes.iter().enumerate() {
                let expected = if i == output { 1.0 } else { 0.0 };
                assert!(
                    (x - Complex64::new(expected, 0.0)).norm() < 1e-12,
                    "input {input:04b}, index {i}: got {x}"
                );
            }
        }
    }
}
