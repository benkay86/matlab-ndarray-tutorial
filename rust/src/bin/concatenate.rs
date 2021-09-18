//! Making arrays bigger.

extern crate blas_src;
use ndarray::{array, concatenate, stack, Array, Axis};

fn main() {
    // Generate some matrices to work with.
    //
    // ```matlab
    // mat1 = [1,2,3; 4,5,6]
    // mat2 = [7, 8, 9]
    // mat3 = [10; 11]
    // ```
    let mat1 = array![[1., 2., 3.], [4., 5., 6.]];
    let mat2 = array![[7., 8., 9.]];
    let mat3 = array![[10.], [11.]];
    println!("Array 1:\n{:?}", mat1);
    println!("Array 2:\n{:?}", mat2);
    println!("Array 3:\n{:?}\n", mat3);

    // In Matlab we can concatenate two matrices together by creating a third
    // matrix containing the first two matrices.  For example, we can
    // concatenate a 1 x 3 matrix onto the bottom of a 2 x 3 matrix to get a
    // 3 x 3 matrix:
    //
    // ```
    // mat1 = [1,2,3; 4,5,6];
    // mat2 = [7,8,9];
    // mat = [mat1; mat2]
    // % 1     2     3
    // % 4     5     6
    // % 7     8     9
    // ```
    //
    // We can do the same with Rust's ndarray using the `ndarray::concatenate()`
    // function.  The arguments are which axis we want to concatenate along
    // (0 to add on a row, 1 to add on a column) and the arrays we want to
    // concatenate.  We'll get an error if the dimensions aren't compatible.
    //
    // ```matlab
    // mat = [mat1; mat2] % concatenate rows
    // mat = [mat1, mat3] % concatenate columns
    // ```
    let mat = concatenate(Axis(0), &[mat1.view(), mat2.view()]).unwrap();
    println!("Concatenate arrays 1 and 2 (rows):\n{:?}", mat);
    // There's also a macro for this if typing `view()` and `unwrap()` is making
    // your fingers tired.
    let mat = concatenate![Axis(1), mat1, mat3];
    println!("Concatenate arrays 1 and 3 (columns):\n{:?}", mat);
    // Studying the above closely, we see that `concatenate` is taking read-only
    // view of the arrays.  That means, best case scenario, at least one copy of
    // mat, mat1, and mat2 (or mat3) exists in memory at the same time.  This is
    // potentially quite wasteful if we are working with big arrays!  So, if you
    // can write code that avoids concatenating altogether, that's best.  For
    // example, if we know the size of the final array we want in advance, then
    // it's most efficient to allocate that array at the start and assign values
    // into it as we go.
    //
    // When concatenation is truly necessary, we may be able to do it more
    // efficienty with `ArrayBase::append()`.  Ideally this will re-use the
    // memory of the first array and copy only the second array, but whether or
    // not this can be done efficiently depends on the layout of the underlying
    // array.  Appending a row onto a row-major (C-order) memory layout array is
    // efficient.  Appending a column onto a row-major array will require
    // reallocations and copies.
    let mut mat = mat1.clone(); // need mat1 for a later example
    mat.append(Axis(0), mat2.view()).unwrap();
    println!("Appending array 2 onto array 1 (rows):\n{:?}", mat);
    // For really fine control we can convert to and from `Vec`.  The same
    // caveats about memory layout apply.  This really isn't necessary since
    // ndarray 0.15 added the `ArrayBase::append()` method.
    //
    // First we convert to raw vectors.  This does not invole any reallocation.
    let mut vec1 = mat1.into_raw_vec();
    let mut vec2 = mat2.into_raw_vec();
    // Now we append to vec1 by moving elements out of vec2.
    // This *will* require a reallocation of vec1 and a memcpy() of vec2, and it
    // *may* require a memcpy() of vec1 if it is reallocated into a new starting
    // address, but if we are lucky then the memory block for vec1 is simply`
    // extended.
    vec1.append(&mut vec2);
    // Drop the empty shell of vec2.
    std::mem::drop(vec2);
    // Convert back to a matrix of the appropriate shape.
    let mat = Array::from_shape_vec((3, 3), vec1).unwrap();
    println!("Append array2 onto array 1 onto array 2 as vector:\n{:?}\n",mat);
    // Unlike as with `concatenate()`, which makes copies, `mat1` and `mat2` no
    // longer exist now because they were (efficiently) consumed/moved by the
    // above procedure.

    // In Matlab we can also "stack" vectors to make a 2d matrix, or 2d matrices
    // to make a 3d matrix.
    //
    // ```
    // mat = [1,2,3]
    // mat = [mat; mat] % stack 2 row vectors into 2d matrix
    // mat(:,:,2) = mat % stack two 2d matrices into 3d matrix
    // ```
    let mat = array![1, 2, 3]; // 1d array
    println!("1d array:\n{:?}", mat);
    // We can stack using the ndarray::stack() function, similar to concatenate.
    let mat = stack(Axis(0), &[mat.view(), mat.view()]).unwrap();
    println!("Stacked to make a 2d array:\n{:?}", mat);
    // Or to save a little bit of typing, with the macro ndarray::stack!().
    let mat = stack!(Axis(0), mat, mat);
    println!("Stacked to make a 3d array:\n{:?}", mat);
}
