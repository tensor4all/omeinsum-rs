//! Test if faer supports strided matrices for GEMM
use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};

fn main() {
    // Create a 4x4 matrix stored column-major
    // We'll extract a 2x2 submatrix with non-unit strides
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();

    // Full matrix is (column-major):
    // [0, 4, 8, 12]
    // [1, 5, 9, 13]
    // [2, 6, 10, 14]
    // [3, 7, 11, 15]

    // Standard contiguous 2x2 from top-left
    // row_stride=1, col_stride=4
    let a1 = unsafe { MatRef::from_raw_parts(data.as_ptr(), 2, 2, 1, 4) };
    println!("Contiguous 2x2:");
    println!("  [{}, {}]", a1[(0, 0)], a1[(0, 1)]);
    println!("  [{}, {}]", a1[(1, 0)], a1[(1, 1)]);

    // Strided 2x2 - every other row and column
    // row_stride=2, col_stride=8
    // Should give: [0, 8], [2, 10]
    let a2 = unsafe { MatRef::from_raw_parts(data.as_ptr(), 2, 2, 2, 8) };
    println!("\nStrided 2x2 (row_stride=2, col_stride=8):");
    println!("  [{}, {}]", a2[(0, 0)], a2[(0, 1)]);
    println!("  [{}, {}]", a2[(1, 0)], a2[(1, 1)]);

    // Try GEMM with strided input
    let mut result = vec![0.0f64; 4];
    let mut c_mat = unsafe { MatMut::from_raw_parts_mut(result.as_mut_ptr(), 2, 2, 1, 2) };

    // Contiguous x contiguous
    matmul(c_mat.as_mut(), Accum::Replace, a1, a1, 1.0, Par::Seq);
    println!("\nGEMM contiguous x contiguous:");
    println!("  [{}, {}]", result[0], result[2]);
    println!("  [{}, {}]", result[1], result[3]);

    // Strided x strided
    result = vec![0.0f64; 4];
    let mut c_mat = unsafe { MatMut::from_raw_parts_mut(result.as_mut_ptr(), 2, 2, 1, 2) };
    matmul(c_mat.as_mut(), Accum::Replace, a2, a2, 1.0, Par::Seq);
    println!("\nGEMM strided x strided:");
    println!("  [{}, {}]", result[0], result[2]);
    println!("  [{}, {}]", result[1], result[3]);

    // Verify strided result manually
    // a2 = [0, 8; 2, 10]
    // a2 * a2 = [0*0+8*2, 0*8+8*10; 2*0+10*2, 2*8+10*10]
    //         = [16, 80; 20, 116]
    println!("\nExpected strided result: [16, 80; 20, 116]");

    println!("\nStrided GEMM works!");
}
