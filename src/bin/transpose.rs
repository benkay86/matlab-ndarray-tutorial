//! Reshaping, transposing, or broadcasting without reallocating.

extern crate blas_src;
use ndarray::{array, Array, Axis};

fn main() {
    // Generate a matrix for us to transpose.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("2x3 matrix:\n{:?}", mat);
    // We can "permute" (not in the statistical sense) the axes of an array.
    // There are several ways of doing this.  Starting with the most general:
    //
    // ```matlab
    // mat = permute(mat, [2,1]) % make axis 1 into axis 2 and 2 into 1
    // ```
    let mut mat = mat.permuted_axes([1, 0]);
    println!("Permuted axes with vector [1, 0]:\n{:?}", mat);
    // Less generally, we can swap any pair of axes.
    mat.swap_axes(0, 1);
    println!("Swapped axes 0 and 1:\n{:?}", mat);
    // Even less generally, when there are only two axes we can reverse them.
    let mat = mat.reversed_axes();
    println!("Reveresed axes:\n{:?}", mat);
    // This last option is the same as transposing a matrix.  As in Matlab,
    // there is a shorthand for transposition in ndarray.
    //
    // ```matlab
    // mat = mat'
    // ```
    let mat = mat.t();
    println!("Transpose of 2d matrix:\n{:?}\n", mat);

    // But beware, transposing a 1d array has no effect.
    let mat = array![1.,2.,3.];
    println!("1d array:\n{:?}", mat);
    let mat = mat.t();
    println!("Transpose of 1d matrix:\n{:?}", mat);
    // However, we can add an axis to convert the 1d array into a 2d array and
    // achieve the same effect as transposing.
    let mat = mat.insert_axis(Axis(1));
    println!("Transpose of 1d matrix by inserting axis:\n{:?}\n", mat);

    // Generate a matrix for us to reshape.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // ```
    let mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("Back to the 2x3 matrix:\n{:?}", mat);
    // In addition to transposing the axes of a matrix, we can reshape it.
    // Recall that arrays are stored as contiguous, linear vectors in memory.
    // We can therefore change the shape of the array by simply changing the
    // shape and stride, without needing to reallocate the actual data.
    //
    // ```matlab
    // mat = reshape(mat, [3,2]) % [1,2; 3,4; 5,6]
    // ```
    //
    // Reshaping might fail if `mat` is not contiguous in memory or if we ask
    // for a shape that is different than the number of elements in `mat`.  In
    // this example we simply `unwrap()` the `Result`.
    let mat = mat.into_shape((3,2)).unwrap();
    println!("Reshaped into 3x2 matrix:\n{:?}", mat);
    // Since an `Array` is pretty much just a fancy `Vec` with shape and stride
    // information, we can also convert to and from a `Vec` without
    // reallocating or copying the data.  Howvever, do beware that the memory
    // layout (row- vs column-order) matters!  Everything in ndarray is row-
    // major by default, so we will get:
    let mat = mat.into_raw_vec(); // to vector [1, 2, 3, 4, 5, 6]
    println!("Underlying raw vector storage:\n{:?}", mat);
    // We can convert back to a 1d array [1, 2, 3, 4, 5, 6]
    let mat = Array::from(mat);
    println!("1d array from raw vector:\n{:?}", mat);
    // And then reshape back to the original 2x3 array:
    // [[1, 2, 3]
    //  [4, 5, 6]]
    let mat = mat.into_shape((2,3)).unwrap();
    println!("Reshaped back to 2x3 matrix:\n{:?}", mat);
    // Or we can reshape directly from a vector in one step.
    let mat = mat.into_raw_vec(); // back to a vector
    let mat = Array::from_shape_vec((2,3), mat).unwrap();
    println!("Reshaped to 2x3 matrix directly from raw vector:\n{:?}\n", mat);

    // With ndarray we can "broadcast" a small array into a larger array without
    // reallocating memory.  Matlab can do this too, but has to allocate memory
    // for the larger array.
    //
    // ```matlab
    // mat = [1,2]
    // bigmat = repmat(mat, 3, 1) % [1,2; 1,2; 1,2]
    // ```
    let mat = array![1, 2];
    println!("A small array:\n{:?}", mat);
    // The syntax in ndarray is a little different from Matlab.  Instead of
    // specifying the number of times to repeat the array along each axis, we
    // specify what we want the final shape to be (3 rows, 2 columns).  The
    // operation will return `None` if we ask for an impossible shape.
    let bigmat = mat.broadcast((3,2)).unwrap();
    println!("Broadcast 3 times along row axis and once along column axis:\n{:?}", bigmat);
    // When broadcasting, the shape of the last axis must remain the same.
    // We can broadcast a 1d matrix of length 2 to an n x 2 matrix.
    // We can broadcast a 3 x 2 matrix to an n x 3 x 2 matrix.
    // But we cannot broadcast a 1d matrix of length 2 to a 2 x n matrix.
    // To achieve the same effect we can broadcast to n x 2 and then take the
    // transpose.
    // let bigmat = mat.broadcast((2,3)).unwrap(); // panics at runtime
    let bigmat = mat.broadcast((3,2)).unwrap().reversed_axes();
    println!("Broadcast and transpose:\n{:?}", bigmat);
    // As with slicing, we can convert the result of any of the above operations
    // into an owned array which is contiguous in memory.
    let bigmat = bigmat.to_owned();
    println!("Owned copy of big array:\n{:?}", bigmat);
}
