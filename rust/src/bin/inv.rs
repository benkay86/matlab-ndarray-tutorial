//! Solve non-singular systems of linear equations and compute matrix inverse.

extern crate blas_src;
use ndarray::{array, Array};
use ndarray_linalg::solve::{Factorize, Inverse, Solve};

fn main() {
    // Tolerance "epsilon" to use in comparisons.
    const EPS: f64 = f64::EPSILON * 8.;

    // Generate a system of linear equations to solve.
    //
    // ```matlab
    // a = [2, 7, 4; 9, 1, 6; 5, 3, 2]
    // b = [1; 2; 3]
    // % Compute the inverse of a.
    // a_inv = inv(a)
    // % Solve for x in a*x = b
    // x = a_inv * b
    // ```
    let a = array![[2., 7., 4.], [9., 1., 6.], [5., 3., 2.]];
    let b = array![1., 2., 3.];
    println!("a =\n{:?}\nb = {:?}", a, b);
    // Invert `a` using the ndarray_linalg::solve::Inverse convenience method.
    // This will actuall perform LU factorization and compute the inverse from
    // it.  Note that we will get a runtime error if `a` is singular.
    let a_inv = a.inv().unwrap();
    println!("a_inv =\n{:?}", a_inv);
    // Let's prove to ourselves that we actually have the inverse.
    // This should print out the identity matrix, or something very close to it
    // within floating point precision.
    println!("a.dot(&a_inv) =\n{:?}", a.dot(&a_inv));
    assert!(Array::<f64, _>::eye(3).abs_diff_eq(&a.dot(&a_inv), EPS));
    // Knowing the inverse of `a` we can solve for `x` in `b = a.dot(&x)`.
    let x = a_inv.dot(&b);
    println!("x = {:?}", x);
    // Prove that it works.
    println!("a.dot(&x) = {:?}\n", a.dot(&x));
    assert!(b.abs_diff_eq(&a.dot(&x), EPS));

    // We do not actually have to compute the inverse of `a` to solve the
    // system.  We can use the LU factorization of `a` directly.
    //
    // ```matlab
    // % TODO
    // ```
    let x = a.solve(&b).unwrap();
    println!("x = {:?}", x);
    // Prove that it works.
    println!("a.dot(&x) = {:?}", a.dot(&x));
    assert!(b.abs_diff_eq(&a.dot(&x), EPS));
    // Suppose we want to solve for multiple x1, x2, etc with multiple b1, b2,
    // etc for the same `a`.  Then it will be most efficient to factorize `a`
    // and resuse the LU factorization for each solution.
    let lu_factorized = a.factorize().unwrap();
    let x = lu_factorized.solve(&b).unwrap();
    println!("x = {:?}", x);
    println!("a.dot(&x) = {:?}", a.dot(&x));
    assert!(b.abs_diff_eq(&a.dot(&x), EPS));
    // And with a different b.
    let b2 = array![4., 5., 6.];
    println!("b2 = {:?}", b);
    let x2 = lu_factorized.solve(&b).unwrap();
    println!("x2 = {:?}", x);
    println!("a.dot(&x2) = {:?}\n", a.dot(&x2));
    assert!(b2.abs_diff_eq(&a.dot(&x2), EPS));

    // Note that in the example above, the solution `x` has the same number of
    // elements as `b`.  If we don't need to keep a copy of b around, we can
    // actually reuse b's memory to store x.  Most methods in ndarray_linalg
    // have some kind of variation like this.  The performance/memory savings
    // can be considerable if `b` is large.
    let x = lu_factorized.solve_into(b); // store the solutioni in b's memory
    let x2 = lu_factorized.solve_into(b2); // store the solution in b2's memory
    println!("x = {:?}\nx2 = {:?}", x, x2);
    // At this point b and b2 have been consumed by the `solve_into()` method
    // calls, and their memory has been reused to store x and x2 without any new
    // allocations.
    // Uncomment the following line to get a compiler error.
    // println!("b = {:?}\nb2 = {:?}", b, b2);
}
