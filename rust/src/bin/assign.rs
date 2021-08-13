//! Assign values to the elements of an array.

extern crate blas_src;
use ndarray::{array, s};

fn main() {
    // Generate a matrix for us to assign to.
    // In Rust we must indicate the matrix is mutable using `mut`.
    //
    // ```matlab
    // mat = [1,2,3; 4,5,6]
    // ```
    let mut mat = array![[1., 2., 3.], [4., 5., 6.]];
    println!("Original matrix:\n{:?}\n", mat);

    // Assign to one element at a time.
    //
    // ```matlab
    // mat(1,1) = 21
    // mat(1,2) = 22
    // ```
    mat[[0, 0]] = 11.;
    println!("Matrix after assignment to [0,0]:\n{:?}", mat);
    // Alternative syntax (see `get()` from <index.rs>).
    match mat.get_mut([0, 1]) {
        Some(el) => {
            *el = 22.;
            println!("Matrix after assignment to [0,1]:\n{:?}", mat);
        }
        None => {
            println!("Index [0,1] is out of bounds.");
        }
    }
    print!("\n");

    // Assign to the entire array at once.
    // There are several ways of doing this in Rust.
    //
    // ```matlab
    // mat = [1,3,5; 2,4,6]
    // ```
    //
    // Move assignment.  The original array in `mat` is dropped and the new
    // array is moved into the variable `mat`.
    mat = array![[1., 3., 5.], [2., 4., 6.]];
    println!("Move assignment:\n{:?}", mat);
    // Clone/copy assignment.  The original array in `mat` is dropped and the
    // new array is copied into `mat`.
    mat.assign(&array![[1., 3., 5.], [2., 4., 6.]]);
    println!("Copy assignment with assign():\n{:?}", mat);
    // Clone/copy assignment from a variable.  The `clone()` makes a copy of
    // `other_mat`.  The variable `other_mat` is unchanged after assignment.
    let other_mat = array![[1., 3., 5.], [2., 4., 6.]];
    mat = other_mat.clone();
    println!("Copy assignment from a variable with clone():\n{:?}", mat);
    // Move assignment from a variable.  The variable `other_mat` no longer
    // exists after the assignment.
    mat = other_mat;
    println!("Move assignment from a variable:\n{:?}\n", mat);
    // println!("other_mat no longer exists:\n{:?}", other_mat); // compiler error

    // We can assign to the entire array from something other than an array.
    // The smaller array is "broadcast" to the size of the larger array.
    //
    // ```matlab
    // mat = zeros(size(mat)) + 2 % fill with 2
    // mat = repmat([1,2,3], 2, 1) % assign [1,2,3] to each row
    // ```
    //
    // Fill with a scalar.
    mat.fill(2.);
    println!("Fill with scalar 2:\n{:?}", mat);
    // Broadcast 1d array [1,2,3] into a 2x3 matrix.
    mat.assign(&array![1., 2., 3.].broadcast((2, 3)).unwrap());
    println!("Broadcast (explicit) row assignment:\n{:?}", mat);
    // Let ndarray try to infer what shape we want to broadcast to.
    mat.assign(&array![1., 2., 3.]);
    println!("Broadcast (automatic) row assignment:\n{:?}\n", mat);

    // Assign to first column and then the last two columns.
    //
    // ```matlab
    // mat(1, :) = [11; 44]
    // mat(2:end, :) = [22,33; 55,66]
    // ```
    //
    // Recognize that this assignment involves slicing the original array.  We
    // can slice mutably with `slice_mut()` and then perform the assignment
    // with `assign()`.  Note in ndarray 0.15 there is also `assign_to()`.
    let mut view1 = mat.slice_mut(s![.., ..1]);
    view1.assign(&array![[11.], [44.]]);
    println!("Matrix after assigning first colum:\n{:?}", mat);
    // Just as there is `slice()` and `slice_mut()`, there is also `column()`
    // and `column_mut()`.  So we could also have done the following.  Note that
    // `column_mut()` gives a 1d rather than 2d array view.
    let mut view1 = mat.column_mut(0);
    view1.assign(&array![11., 44.]);
    // Or even:
    mat.column_mut(0).assign(&array![11., 44.]);
    // Now let's assign to the last 2 columns.
    let mut view2 = mat.slice_mut(s![.., 1..]);
    view2.assign(&array![[22., 33.], [55., 66.]]);
    println!("Matrix after assigning last two colums:\n{:?}\n", mat);
    // Recall from the Rust borrowing rules that we can have two mutable
    // references at the same time.  That means once we create `view2` we can
    // no longer use `view1`.
    // Uncomment the following line for a compiler error.
    // view1.assign(&array![[111.], [444.,]]);

    // What if we want several mutable references to the same matrix at the same
    // time?  This doesn't come up much in Matlab, except in parfor loops when
    // we get mysterious errors about a variable that cannot be "classified."
    // But it will come up in Rust/ndarray a lot, *especially* when we get into
    // parfor and other parallel/multi-threaded programming techniques!
    //
    // In the above example it would actually be OK to have mutable view into
    // the first column and last 2 columns at the same time because those views
    // are disjoint, i.e. they don't overlap.  We can express this expliticly
    // with `multi_slice_mut()`, here taking a view for each column.
    let (mut col1, mut col2, mut col3) = mat.multi_slice_mut((
        s![.., 0],   // 1st column as 1d view
        s![.., 1],   // 2nd column as 1d view
        s![.., 2..], // 3rd column as 2d column view (just for fun)
    ));
    col1.fill(1.); // fill 1st column with ones
    col2.assign(&array![2., 22.]); // assign 1d array to 2nd column
    col3.assign(&array![[3.], [33.]]); // assign 2d column vector to 3rd column
    println!("Matrix after assigning to multiple slices:\n{:?}", mat);
}
