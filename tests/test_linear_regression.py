from regressions import linear_regression as lr
import numpy as np
import pytest

#Sample data
x_data = [7.52454557, 4.73430351, 5.36380876, 5.18944064, 8.11937597, 6.92708317, 5.44973327, 4.68660221, 9.90192846, 3.85584356]
y_data = [23.16013386, 15.84396669, 17.24982032, 16.63386921, 24.52164639, 21.24304725, 17.73245848, 15.58223633, 30.50417894, 13.17874506]
x = np.array(x_data)
y = np.array(y_data)


''' Unit tests for error_function '''


#Test if the function raises ValueError if one of the arrays is empty
def test_error_function_one_array_empty():
    with pytest.raises(ValueError) as er:
        lr.error_function(0,0,np.array([]), np.array([1]))

    assert str(er.value) == 'Arrays cannot be empty'

    with pytest.raises(ValueError) as er:
        lr.error_function(0, 0, np.array([1, 4, 6, 4]), np.array([]))

    assert str(er.value) == 'Arrays cannot be empty'


#Test if the function raises ValueError if an array contains not numeric value
def test_gradient_desc_not_numeric_array():
    with pytest.raises(ValueError) as er:
        lr.error_function(0, 0, np.array([1, 4, 6, 4]), np.array([2, 3, 1, 'a']))

    assert str(er.value) == "Array contains non-numeric values"

    with pytest.raises(ValueError) as er:
        lr.error_function(0, 0, np.array([1, 'b', 6, 4]), np.array([4, 4, 3, 1]))


#Test if the function raises ValueError if arrays have a different lengths
def test_error_function_not_equel_arrays_len():
    with pytest.raises(ValueError) as er:
        lr.error_function(0,0,np.array([2,4,5]), np.array([1,0]))

    assert str(er.value) == 'The length of the "x" array must match the length of the "y" array'


#Test if the functions raises TypeError if data arrays are not numpy arrays
def test_error_function_arrays_not_numpy():
    with pytest.raises(TypeError) as er:
        lr.error_function(0,0, 'a', np.array([1,0]))

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.error_function(0,0, [3, 2, 1], np.array([1,0]))

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.error_function(0, 0, np.array([1, 0]), 34.2)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.error_function(0, 0, np.array([1, 0]), [5,3,2,1])

    assert str(er.value) == 'The data must be passed as a numpy array'


#Test if the function raises TypeError if 'm' or 'b' are not numerical values
def test_error_function_mb_not_a_number():
    with pytest.raises(TypeError) as er:
        lr.error_function('a', 0, np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]))

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.error_function(3.124, [5, 2], np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]))

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.error_function('m', 'b', np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]))

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"


#Test the function with valid inputs and known expected outputs
def test_error_function_valid():
    assert lr.error_function(3.0654156373257058, 0.4883756479577883, x, y) == 0.36782329429247584
    assert lr.error_function(2.9647999931682767, 1.1633863098122286, x, y) == 0.20578109861690871
    assert lr.error_function(2.6981780805793, 2.9460358145416383, x, y) == 0.11586958157175424


''' Unit tests for gradient_desc '''


#Test if the function raises ValueError if one of the arrays is empty
def test_gradient_desc_one_array_empty():
    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0,0,np.array([]), np.array([1]), 0.0001)

    assert str(er.value) == 'Arrays cannot be empty'

    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0, 0, np.array([1, 4, 6, 4]), np.array([]), 0.0001)

    assert str(er.value) == 'Arrays cannot be empty'


#Test if the function raises ValueError if an array contains not numeric value
def test_gradient_desc_not_numeric_array():
    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0, 0, np.array([1, 4, 6, 4]), np.array([5, 3, 2, 'a']), 0.0001)

    assert str(er.value) == "Array contains non-numeric values"

    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0, 0, np.array([1, 4, 6, 'b']), np.array([4, 2, 1,4]), 0.0001)


#Test if the function raises ValueError if arrays have a different lengths
def test_gradient_desc_not_equel_arrays_len():
    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0,0,np.array([2,4,5]), np.array([1,0]), 0.0001)

    assert str(er.value) == 'The length of the "x" array must match the length of the "y" array'


#Test if the functions raises TypeError if data arrays are not numpy arrays
def test_gradient_desc_arrays_not_numpy():
    with pytest.raises(TypeError) as er:
        lr.gradient_desc(0,0, 'a', np.array([1,0]), 0.0001)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.gradient_desc(0,0, [3, 2, 1], np.array([1,0]), 0.0001)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.gradient_desc(0, 0, np.array([1, 0]), 34.2, 0.0001)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.gradient_desc(0, 0, np.array([1, 0]), [5,3,2,1], 0.0001)

    assert str(er.value) == 'The data must be passed as a numpy array'


#Test if the function raises TypeError if 'm' or 'b' are not numerical values
def test_gradient_desc_mb_not_a_number():
    with pytest.raises(TypeError) as er:
        lr.gradient_desc('a', 0, np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]), 0.0001)

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.gradient_desc(3.124, [5, 2], np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]), 0.0001)

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.gradient_desc('m', 'b', np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]), 0.0001)

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"


#Test if the function raises ValueError if learning rate in not greater than or equal to 0 or is not a numerical value
def test_gradient_desc_learning_rate_invalid():
    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0, 0, np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]), 'a')

    assert str(er.value) == 'The learning rate must be greater than or equal to 0 and must be passed as an int or a float'

    with pytest.raises(ValueError) as er:
        lr.gradient_desc(0, 0, np.array([2, 3, 1]), np.array([1, 0.5, 0.32145]), -0.1)

    assert str(er.value) == 'The learning rate must be greater than or equal to 0 and must be passed as an int or a float'


#Test the function with valid inputs and known expected outputs
def test_gradient_desc_valid():
    assert lr.gradient_desc(0, 0, x, y, 0.0001) == (0.025938159090395704, 0.0039130020506)
    assert lr.gradient_desc(7.52454557, 4.73430351, x, y, 0.005) == (5.4197940230437975, 4.4179498346656105)
    assert lr.gradient_desc(2, 3, x, y, 1) == (59.041411235506956, 11.428954458)


''' Unit tests for train_linear_regression '''


#Test if the function raises ValueError if one of the arrays is empty
def test_train_linear_regression_one_array_empty():
    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([]), np.array([1]), 100, 0.0001, 0, 0)

    assert str(er.value) == 'Arrays cannot be empty'

    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([1]), np.array([]), 100, 0.0001, 0, 0)

    assert str(er.value) == 'Arrays cannot be empty'


#Test if the function raises ValueError if an array contains not numeric value
def test_train_linear_regression_not_numeric_array():
    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([1, 3, 4]), np.array([1, 2, 'a']), 100, 0.0001, 0, 0)

    assert str(er.value) == "Array contains non-numeric values"

    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([1, 3, 'b']), np.array([1, 2, 4]), 100, 0.0001, 0, 0)

    assert str(er.value) == "Array contains non-numeric values"


#Test if the function raises ValueError if arrays have a different lengths
def test_train_linear_regression_not_equel_arrays_len():
    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([2, 5, 10.4324231]), np.array([1.432]), 100, 0.0001, 0, 0)

    assert str(er.value) == 'The length of the "x" array must match the length of the "y" array'


#Test if the functions raises TypeError if data arrays are not numpy arrays
def test_train_linear_regression_arrays_not_numpy():
    with pytest.raises(TypeError) as er:
        lr.train_linear_regression('a', np.array([1]), 100, 0.0001, 0, 0)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.train_linear_regression(np.array([1]), 433.2, 100, 0.0001, 0, 0)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.train_linear_regression([2, 3, 4], np.array([1]), 100, 0.0001, 0, 0)

    assert str(er.value) == 'The data must be passed as a numpy array'

    with pytest.raises(TypeError) as er:
        lr.train_linear_regression(np.array([2, 3.22]), [1, 34, 3.35], 100, 0.0001, 0, 0)

    assert str(er.value) == 'The data must be passed as a numpy array'


#Test if the function raises TypeError if 'm' or 'b' are not numerical values
def test_train_linear_regression_mb_not_a_number():
    with pytest.raises(TypeError) as er:
        lr.train_linear_regression(np.array([2]), np.array([1]), 100, 0.0001, 'a', 0)

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.train_linear_regression(np.array([3]), np.array([1]), 100, 0.0001, 0, [3, 2.34, 4])

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.train_linear_regression(np.array([1]), np.array([1]), 100, 0.0001, 'a', 'b')

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"


#Test if the function raises ValueError if learning rate in not greater than or equal to 0 or is not a numerical value
def test_train_linear_regression_learning_rate_invalid():
    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([2]), np.array([1]), 100, 'a', 0, 0)

    assert str(er.value) == 'The learning rate must be greater than or equal to 0 and must be passed as an int or a float'

    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([3]), np.array([1]), 100, -0.0001, 0, 0)

    assert str(er.value) == 'The learning rate must be greater than or equal to 0 and must be passed as an int or a float'


#Test if the function raises ValueError if 'epochs' is not greater than or equal to 0 or is not an 'int'
def test_train_linear_regression_epochs_invalid():
    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([3]), np.array([1]), 0.2, 0.0001, 0, 0)
    assert str(er.value) == 'The epochs must be greater than or equal to 0 and must be passed as an int.'

    with pytest.raises(ValueError) as er:
        lr.train_linear_regression(np.array([3]), np.array([1]), -102, 0.0001, 0, 0)
    assert str(er.value) == 'The epochs must be greater than or equal to 0 and must be passed as an int.'


#Test the function with valid inputs and known expected outputs
def test_train_linear_regression_valid():
    assert lr.train_linear_regression(x, y, 1000, 0.0001, 0, 0) == (3.0654156373257058, 0.4883756479577883)
    assert lr.train_linear_regression(x, y, 300, 0.001, 4, 3) == (2.721581378957579, 2.7892898081508193)
    assert lr.train_linear_regression(x, y, 400, 0.005, 3, 3) == (2.7191923774907503, 2.805246750951246)


''' Unit tests for predict '''


#Test if the function raises TypeError if 'm' or 'b' are not numerical values
def test_predict_mb_not_a_number():
    with pytest.raises(TypeError) as er:
        lr.predict(2, 'a', np.array([2.21, 3.12, 4.12]))

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"

    with pytest.raises(TypeError) as er:
        lr.predict([1,2,77], 5, np.array([2.21, 3.12, 4.12]))

    assert str(er.value) == "The coefficients 'm' and 'b' must be numeric"


#Test if the function raises ValueError if an array contains not numeric value
def test_predict_not_numeric_array():
    with pytest.raises(ValueError) as er:
        lr.predict(2, 1, np.array([2.21, 3.12, 4.12, 'a']))

    assert str(er.value) == "Array contains non-numeric values"


#Test if the function raises TypeError if 'x' is not numpy array or numerical value
def test_predict_x_data_type():
    with pytest.raises(TypeError) as er:
        lr.predict(0, 0, 'a')

    assert str(er.value) == "The input variable 'x' must be an integer, float, or a numpy array"


#Test the function with valid inputs and known expected outputs
def test_predict_valid():
    assert lr.predict(2.721581378957579, 2.7892898081508193, 12) == 35.44826635564177
    assert np.allclose(lr.predict(2.721581378957579, 2.7892898081508193, np.array([2, 5, 8, 3, 11, 44, 23, 1])),
            np.array([  8.23245257,  16.3971967,   24.56194084,  10.95403395,  32.72668498,
            122.53887048,  65.38566152,   5.51087119]))