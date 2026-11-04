"""
EECS 445 - Introduction to Machine Learning
Fall 2026  - Project 2

Test Image Standardizer
    Test the ImageStandardizer class's fit and transform functions.
    Usage: python test_ImageStandardizer.py
"""

import numpy as np

from dataset import ImageStandardizer


def test_image_standardizer():
    """Test dataset.ImageStandardizer.fit() and dataset.ImageStandardizer.transform() functions."""
    standardizer = ImageStandardizer()
    
    # Test Case 1
    X1 = np.array([[[[1, 2, 3], [4, 5, 6]],
                   [[7, 8, 9], [10, 11, 12]]]], dtype=np.float64)
    expected_mean1 = np.array([5.5, 6.5, 7.5], dtype=np.float64)
    expected_std1 = np.array([3.35410197, 3.35410197, 3.35410197], dtype=np.float64)
    
    standardizer.fit(X1)
    assert np.allclose(standardizer.image_mean, expected_mean1), "Test case 1 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std1), "Test case 1 failed on std"

    X1_transformed_expected = np.array(
        [[[[-1.34164079, -1.34164079, -1.34164079],
        [-0.4472136,  -0.4472136,  -0.4472136 ]],
        [[ 0.4472136,   0.4472136,   0.4472136 ],
        [ 1.34164079,  1.34164079,  1.34164079]]]], dtype=np.float64)
    X1_transformed_actual = standardizer.transform(X1)
    assert np.allclose(X1_transformed_actual, X1_transformed_expected), "Test case 1 failed on transform"
    
    # Test Case 2
    X2 = np.array([[[[1], [3]], [[5], [7]]]], dtype=np.float64)
    expected_mean2 = np.array([4], dtype=np.float64)
    expected_std2 = np.array([[2.23606798]], dtype=np.float64)
    
    standardizer.fit(X2)
    assert np.allclose(standardizer.image_mean, expected_mean2), "Test case 2 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std2), "Test case 2 failed on std"

    X2_transformed_expected = np.array(
        [[[[-1.34164079],
        [-0.4472136 ]],
        [[ 0.4472136 ],
        [ 1.34164079]]]])
    X2_transformed_actual = standardizer.transform(X2)
    assert np.allclose(X2_transformed_actual, X2_transformed_expected), "Test case 2 failed on transform"
    
    # Test Case 3
    X3 = np.array([[[[5, 5, 5], [5, 5, 5]],
                   [[5, 5, 5], [5, 5, 5]]]], dtype=np.float64)
    expected_mean3 = np.array([5, 5, 5], dtype=np.float64)
    expected_std3 = np.array([0, 0, 0], dtype=np.float64)
    
    standardizer.fit(X3)
    assert np.allclose(standardizer.image_mean, expected_mean3), "Test case 3 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std3), "Test case 3 failed on std"
    
    # Test Case 4
    X4 = np.array([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
        [[[9, 10], [-11, 12]], [[13, 14], [15, 16]]],
        [[[17, -18], [19, 20]], [[21, 22], [-23, 24]]]
    ], dtype=np.float64)
    expected_mean4 = np.array([6.33333333, 10], dtype=np.float64)
    expected_std4 = np.array([12.31079021, 10.8012345], dtype=np.float64)

    standardizer.fit(X4)
    assert np.allclose(standardizer.image_mean, expected_mean4), "Test case 4 failed on mean"
    assert np.allclose(standardizer.image_std, expected_std4), "Test case 4 failed on std"


if __name__ == "__main__":
    test_image_standardizer()
    print("All tests passed!")
