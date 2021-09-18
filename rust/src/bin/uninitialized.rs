//! Create (allocate memory for) a matrix and initialize it later.
//!
//! This is an advanced topic.  Make sure you understand the implications of
//! using [unsafe](https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html) and
//! are familiar with the [dangers of unitialized memory](https://doc.rust-lang.org/std/mem/union.MaybeUninit.html)
//! before you proceed.

#![allow(unused_imports)]
extern crate blas_src;
use core::mem::MaybeUninit;
use ndarray::{s, Array};

fn main() {
    // In Matlab we often pre-allocate a matrix and then initialize it later
    // with some non-trivial computational operation.
    //
    // ```matlab
    // mat = zeros(3, 2);
    // for i=1:3
    //     for j=1:2
    //         % Initialize with (row-major) offset.
    //         mat(i, j) = 2 * (i - 1) + (j - 1);
    //     end
    // end
    // ```
    let mut mat = Array::<usize, _>::zeros((3, 2));
    for ((i, j), el) in mat.indexed_iter_mut() {
        *el = 2 * i + j;
    }
    println!("mat =\n{:?}", mat);

    // You will note that, technically, each element of the array was
    // initialized twice: first with 0, and then with its intended value.
    // Wouldn't it be more efficient to initialize each element just once?
    // Ndarray gives us many flexible array initialization methods for this.
    let mat = Array::from_shape_fn((3, 2), |(x, y)| 2 * x + y);
    println!("mat =\n{:?}", mat);

    // What if setting the values in the array is very non-trivial and can't be
    // done with methods like from_shape_fn()?  In this case it is still quite
    // fast to initialize an array with zeros(), and then initialize a seond
    // time in a for loop as above.
    // But if you are writing performance-critical code and can't afford to
    // double-initialize, the following method allocates memory for the array
    // *without* initializing that memory.
    let mut mat = Array::<usize, _>::uninit((3, 2));
    // We can't *read* from the newly-allocated memory because it has not been
    // initialized, but it is totally safe to *write* to it.
    for ((i, j), el) in mat.indexed_iter_mut() {
        *el = MaybeUninit::new(2 * i + j);
    }
    // Now comes the dangerous part.  We must guarantee to the compiler that
    // *all* the memory in the array has been initialized (written to) before we
    // can read from *any* of that memory.  Note the use of unsafe.
    let mat = unsafe { mat.assume_init() };
    // In this case we *did* correctly initialize all the memory and the above
    // `unsafe` is kosher.  Printing out the matrix will work fine.
    println!("mat =\n{:?}", mat);

    // The following examples will compile and probably even run if you
    // uncomment them, but they do not satisfy the preconditions of unsafe and
    // would likely cause difficult-to-debug problems in a more complex program.
    // Unsafe is tricky, be careful with it!

    // This would be bad because we didn't initialize all of the array's memory.
    // Only the first element gets initialized.
    //
    // let mut mat = Array::<usize, _>::uninit((3, 2));
    // mat[[0,0]] = MaybeUninit::new(0);
    // let mat = unsafe { mat.assume_init() };
    // println!("mat =\n{:?}", mat);

    // This kind of looks like it's OK, but it's not.
    // The slice_move() does not return an array with a contiguous memory layout
    // in this case.  Therefore, the for loop will not initialize all of the
    // array's memory (only the part that is sliced).  Since *all* of the memory
    // was not initialized, it isn't safe to call assume_init().
    //
    // let mat = Array::<usize, _>::uninit((3, 2));
    // let mut mat = mat.slice_move(s![..;2, ..]); // take every other row
    // for ((i,j), el) in mat.indexed_iter_mut() {
    //     *el = MaybeUninit::new(2 * i + j);
    // }
    // let mat = unsafe { mat.assume_init() };
    // println!("mat =\n{:?}", mat);
}
