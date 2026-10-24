# Schrodinger
Schrodinger is an intentionally-inefficient pedagogical schrodinger-style state-vector simulation for quantum circuits. During the simulation, each gate operation is a) executed in serial and b) calculated by forming the entire 2^n * 2^n Kronecker product to apply the gate to the state vector. 

# Examples
## [Teleportation](https://github.com/DanielRJohnson/Schrodinger/blob/main/examples/teleportation.rs)
The classic Alice-to-Bob quantum state teleportation problem (but reduced to classical inputs and outputs)
```
λ cargo run --example teleportation -- 0
Before: Alice has 0. After: Bob has 0
λ cargo run --example teleportation -- 1
Before: Alice has 1. After: Bob has 1
```

## [Half-Adder](https://github.com/DanielRJohnson/Schrodinger/blob/main/examples/quantum_half_adder.rs)
Implementation of a classical half-adder via a quantum circuit.
```
λ cargo run --example quantum_half_adder -- 1 0
1 + 0 = 01 (sum = 1, carry = 0)
```

## [Ripple-Carry Adder](https://github.com/DanielRJohnson/Schrodinger/blob/main/examples/quantum_ripple_carry_adder.rs)
Have you ever wanted to iteratively build and multiply 4096x4096 matrices over and over in order to add two 4-bit numbers? You've come to the right place! (On my Ryzen 7 7700X release takes ~15sec, debug just takes a brisk ~12min!)
```
λ cargo build --release --example quantum_ripple_carry_adder
λ ./target/release/examples/quantum_ripple_carry_adder 15 15
15 + 15 = 30  (4-bit adder, 12 qubits)
```
