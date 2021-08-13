//! Indexing and selecting elements of an array.
//!
//! See <slicing.rs> for more advanced examples.

extern crate blas_src;
use ndarray::{array, Axis};

fn main() {
    // Generate a matrix for us to index in different ways.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("Full 2x3 matrix:\n{:?}\n", mat);

    // Simplest example: Index first (upper left) element in the 2x2 matrix.
    // Note that ndarray counts from zero whereas Matlab counts from 1.
    //
    // ```matlab
    // el = mat(1,1)
    // ```
    let el = mat[[0, 0]];
    println!("First element: {:?}", el);
    // We can also index using the `get()` method.  Both the index operation
    // `[]` and `get()` perform bounds checking.  Indexing with `[]` panics if
    // the index is out of bounds, whereas `get()` returns `None` if the index
    // is out of bounds.
    //
    // ```matlab
    // el = mat(1,3)
    // ```
    match mat.get([0, 2]) {
        Some(el) => println!("Top right element: {:?}\n", el),
        None => println!("Index out of bounds.\n"),
    }

    // We can also select entire rows or columns of a matrix.  This can often
    // be done more efficiently by [slicing](slicing.rs), but selecting lets us
    // choose arbitrary rows/cols whereas slicing does not.
    // `Axis(0)` selects along the first axis (rows) and `Axis(1)` selects along
    // the second axis (columns).
    //
    // ```matlab
    // submat = mat(1,:) % first row
    // submat = mat(:,[2,3]) % last 2 columns
    // ```
    let submat = mat.select(Axis(0), &[0]);
    println!("First row:\n{:?}", submat);
    let submat = mat.select(Axis(1), &[1, 2]);
    println!("Second and third columns:\n{:?}\n", submat);

    // We can also select elements from a vector, or 1d array.  In Rust this is
    // also done with `select()`.  Note the use of `Axis(0)` for a 1d array.
    // Unlike Matlab, which can treat 1d arrays as row or column vectors, Rust
    // treats all 1d arrays as truly one dimensional (only has Axis(0)).
    //
    // ```
    // v = [1, 2, 3]; % or mat1d = [1; 2; 3];
    // submat = mat1d([2,3])
    // ```
    let v = array![1., 2., 3.];
    let submat = v.select(Axis(0), &[1, 2]);
    println!("Second and third columns of first row:\n{:?}\n", submat);

    // Matlab offers us yet another way of indexing multidimensional arrays with
    // a single scalar index (although it's not used much in practice).
    //
    // ```matlab
    // el = mat(2) % 4
    // el = mat(3) % 2
    // ```
    //
    // This scalar indexes the array in contiguous memory order.  We think of
    // matrices as 2-dimensional constructs, but computers have no intrinsic
    // concept of dimensionality and just think of the memory linearly.  In
    // Matlab and in Rust/ndarray, the 2d matrix is laid out as one contiguous
    // block in memory.  In Matlab it is laid out in column-major order:
    // 1 4 2 5 3 6
    // And in Rust it is laid out in row-major order:
    // 1 2 3 4 5 6
    // When we index by a row and column, the computer has stored the shape of
    // the array and uses that to figure out the offset from the beginning of
    // the memory block.  But we can also index by the offset directly.  Note
    // that since Matlab and Rust have different memory layouts, offset indexing
    // will give different results in each case.
    assert!(mat.is_standard_layout()); // verify row-major contiguous layout
    let el = mat.as_slice_memory_order().unwrap()[1];
    println!("Memory offset 1: {:?}", el); // 2, but in Matlab mat(2) is 4
    let el = mat.as_slice_memory_order().unwrap()[2];
    println!("Memory offset 2: {:?}", el); // 3, but in Matlab mat(3) is 2
                                           // If this seems confusing, it is!  And you probably won't want to use
                                           // offsets for indexing very often.
}
