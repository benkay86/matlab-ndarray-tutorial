//! Compare scalars and arrays.

extern crate blas_src;
use approx::AbsDiffEq;
use ndarray::array;

fn main() {
    // Comparing scalars.  Note that, due to floating point error, we often
    // check to see if two floating point numbers are *approximately* equal
    // to within machine precision (epsilon).
    //
    // ```matlab
    // a = 1;
    // b = 2;
    // a == a
    // abs(a - a) < eps(1)
    // a <= a
    // a < a
    // a < b
    // ```
    let a: f64 = 1.;
    let b: f64 = 2.;
    println!("a = {:?}, b = {:?}", a, b);
    println!("a == a: {:?}", a == a);
    println!("a approx a: {:?}", (a - a).abs() < f64::EPSILON);
    println!("a <= a: {:?}", a <= a);
    println!("a < a: {:?}", a < a);
    println!("a < b: {:?}\n", a < b);

    // Comparing arrays.
    //
    // ```
    // a = [1, 2, 3; 4, 5, 6];
    // b = [2, 3, 4; 5, 6, 7];
    // all(all(a == a))
    // all(all(abs(a - a) < eps(1)))
    // all(all(a < b))
    // all(all(a > b))
    // ```
    let a = array![[1., 2., 3.], [4., 5., 6.]];
    let b = array![[2., 3., 4.], [5., 6., 7.]];
    println!("a == a: {:?}", a == a);
    println!("a approx a: {:?}", a.abs_diff_eq(&a, f64::EPSILON));
    println!(
        "a < b: {:?}",
        ndarray::Zip::from(&a).and(&b).all(|a, b| a < b)
    );
    println!(
        "a > b: {:?}",
        ndarray::Zip::from(&a).and(&b).all(|a, b| a > b)
    );
}
