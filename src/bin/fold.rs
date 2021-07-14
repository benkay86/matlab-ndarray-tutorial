//! Fold, or reduce arrays to a single value.

extern crate blas_src;
use ndarray::{array, Axis, Zip};

fn main() {
    // In <for.rs> we saw how to sum the elements of an array using a for
    // loop or using the `sum()` method.  Summing is an example of "folding,"
    // "accumulating," or "reducing" the elements of a vector into a scalar.
    // Ndarray has a special method for this type of iteration.
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("mat = \n{:?}", mat);
    println!("Sum using sum() = {:?}", mat.sum());
    let mut sum = 0.;
    mat.iter().for_each(|el| sum += el);
    println!("Sum using for loop = {:?}", sum);
    // The fold method takes an initial value (zero in this example of
    // summation) and a closure.  The closure takes two arguments, the result
    // of the previous call to the closure and the next element of the array.
    // In each successive call to the closure, `sum` will be the value returned
    // by the previous call to the closure.  The grand total sum is returned at
    // the end.
    //
    // The elements of the array are visited in an arbitrary order.
    println!("sum using fold = {:?}\n", mat.fold(0., |sum, el| sum + el));

    // We can also fold along an axis to generate a new array.
    println!("Sum of rows using fold_axis() =\n{:?}", mat.fold_axis(Axis(1), 0., |sum, col| sum + col));
    println!("Sum of rows using sum_axis() =\n{:?}\n", mat.sum_axis(Axis(1)));

    // We can find the sum of two arrays at once using `Zip` combined with
    // `fold()`.  This is actually *more* efficient than vectorized addition.
    let sum = Zip::from(&mat).and(&mat).fold(0., |sum, el1, el2| sum + el1 + el2);
    println!("Sum of mat + mat using Zip =\n{:?}", sum);
    // This requies an extra heap allocation to store the temporary value of
    // `mat + mat`, making it less efficient than `Zip` above.
    let sum = (&mat + &mat).sum();
    println!("Sum of mat + mat using vectorized addition an =\n{:?}\n", sum);

    // We can use folding for many purposes.  For example, we can check if any
    // of the elements of the array is equal to 2.
    let ans = mat.fold(false, |ans, el| {
        if el == &2. { true }
        else { ans }
    });
    println!("Are any elements equal to 2? {:?}", ans);
    // Note in this case it would have been faster to use a for loop
    // so that we can break early once the condition is matched.  This is called
    // a short-circuiting fold.
    let mut ans = false;
    for el in mat.iter() {
        if el == &2. {
            ans = true;
            break;
        }
    }
    println!("Are any elements equal to 2? {:?}", ans);
    // As we'll see later with parallel iterators, there are ways to "trick" an
    // iterator combinator to short circuit when control-flow loops are not
    // available.  See also:
    // <https://doc.rust-lang.org/core/ops/enum.ControlFlow.html>
    let ans = mat.iter().try_fold(false, |ans, el| {
        if el == &2. { Err(()) } // causes try_fold to return early
        else { Ok(ans) }
    }).unwrap_or(true); // convert `Err(())`` to `true`
    println!("Are any elements equal to 2? {:?}", ans);
    // Of course, as with sum, there is already a built-in folder for this
    // common operation.
    let ans =  mat.iter().any(|el| el == &2.);
    println!("Are any elements equal to 2? {:?}", ans);
    // In addition to `sum()` and `any()`, see `all()`, `product()`, `mean()`,
    // `var()`, `std()`, `min()`, `max()`, and many other build-in folders.
    // Note that some of these methods are implemented on `ArrayBase` directly
    // while others require you to turn it into an `iter()` first.
}
