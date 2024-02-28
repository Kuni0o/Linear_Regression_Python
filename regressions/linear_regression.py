import numpy as np


def error_function(m, b, x, y):
    '''
    Calculate the Mean Squared Error (MSE) between the observed values and the values predicted by a linear regression
    function.

    Parameters
    ----------
    m : float
        Slope of the regression line.

    b : float
        Y-intercept of the regression line.

    x : numpy.ndarray
        Independent variable values from the dataset.

    y : numpy.ndarray
        Observed dependent variable values from the dataset.

    Returns
    -------
    error_sum / n : float
        Average MSE (Mean Squared Error) value between the observed and predicted values.

    Raises
    ------
    ValueError
        If the lengths of the 'x' and 'y' arrays do not match or if either array is empty.

    TypeError
        If the 'x' and 'y' arguments are not numpy arrays, or if the slope and y-intercept are not numeric.

    '''

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('The data must be passed as a numpy array')

    if not np.all(np.isreal(x)) or not np.all(np.isreal(y)):
        raise ValueError("Array contains non-numeric values")

    if len(x) == 0 or len(y) == 0:
        raise ValueError('Arrays cannot be empty')

    if len(x) != len(y):
        raise ValueError('The length of the "x" array must match the length of the "y" array')

    if not isinstance(m, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("The coefficients 'm' and 'b' must be numeric")

    n = len(x)
    error_sum = np.sum((y - ((m * x) + b)) ** 2)

    return error_sum / n



def gradient_desc(m, b, x, y, L):
    '''
    Perform gradient descent to update the coefficients 'm' and 'b' of a linear regression model.

    Parameters
    ----------
    m : float
        Current slope of the regression line.

    b : float
        Current y-intercept of the regression line.

    x : numpy.ndarray
        Independent variable values from the dataset.

    y : numpy.ndarray
        Observed dependent variable values from the dataset.

    L : int or float
        Learning rate for gradient descent. Must be greater than or equal to 0.

    Returns
    -------
    m : float
        Updated slope of the regression line after gradient descent.

    b : float
        Updated y-intercept of the regression line after gradient descent.

    Raises
    ------
    TypeError
        If the coefficients 'm' and 'b' are not numeric, or if the learning rate 'L' is not numeric.

    ValueError
        If the lengths of the 'x' and 'y' arrays do not match, if either array is empty, or if the learning rate is
        negative.

    '''

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('The data must be passed as a numpy array')

    if not np.all(np.isreal(x)) or not np.all(np.isreal(y)):
        raise ValueError("Array contains non-numeric values")

    if len(x) == 0 or len(y) == 0:
        raise ValueError('Arrays cannot be empty')

    if len(x) != len(y):
        raise ValueError('The length of the "x" array must match the length of the "y" array')

    if not isinstance(m, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("The coefficients 'm' and 'b' must be numeric")
    
    if not isinstance(L, (int, float)) or L < 0:
        raise ValueError('The learning rate must be greater than or equal to 0 and must be passed as an int or a float')


    n = len(x)
    m_grad = np.sum((-2/n) * x * (y - (m * x + b)))
    b_grad = np.sum((-2/n) * (y - (m * x + b)))

    m -= (L * m_grad)
    b -= (L * b_grad)

    return m, b



def train_linear_regression(x, y, epochs, learning_rate = 0.0001, m = 0, b = 0):
    '''
    Train a linear regression model using gradient descent.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable values from the dataset.

    y : numpy.ndarray
        Observed dependent variable values from the dataset.

    epochs : int
        Number of training epochs (iterations).

    learning_rate : int or float, optional
        Learning rate for gradient descent. Must be greater than or equal to 0. Default is 0.0001.

    m : float, optional
        Initial slope of the regression line. Default is 0.

    b : float, optional
        Initial y-intercept of the regression line. Default is 0.

    Returns
    -------
    m : float
        Final slope of the regression line after training.

    b : float
        Final y-intercept of the regression line after training.

    Raises
    ------
    TypeError
        If the coefficients 'm' and 'b' are not numeric, or if the learning rate is not numeric.

    ValueError
        If the lengths of the 'x' and 'y' arrays do not match, if either array is empty, if the learning rate is
        negative, or if the 'epochs' is negative or not an int.

    '''

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('The data must be passed as a numpy array')

    if not np.all(np.isreal(x)) or not np.all(np.isreal(y)):
        raise ValueError("Array contains non-numeric values")

    if len(x) == 0 or len(y) == 0:
        raise ValueError('Arrays cannot be empty')

    if len(x) != len(y):
        raise ValueError('The length of the "x" array must match the length of the "y" array')

    if not isinstance(m, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("The coefficients 'm' and 'b' must be numeric")
    
    if not isinstance(learning_rate, (int, float)) or learning_rate < 0:
        raise ValueError('The learning rate must be greater than or equal to 0 and must be passed as an int or a float')

    if epochs < 0 or not isinstance(epochs, int):
        raise ValueError('The epochs must be greater than or equal to 0 and must be passed as an int.')

    for i in range(epochs):
        if i % 50 == 0:
            print(f"{i}. loss: {error_function(m, b, x, y)}")
        m, b = gradient_desc(m, b, x, y, learning_rate)
    
    return m, b



def predict(m, b, x):
    '''
    Predict the dependent variable values based on the independent variable values and the coefficients of a linear
    regression model.

    Parameters
    ----------
    m : float
        Slope of the regression line.

    b : float
        Y-intercept of the regression line.

    x : int, float, or numpy.ndarray
        Independent variable value or array of independent variable values.

    Returns
    -------
    np.ndarray
        Predicted dependent variable values.

    Raises
    ------
    TypeError
        If the coefficients 'm' and 'b' are not numeric, or if the independent variable values 'x' are not numeric or
        convertible to a numpy array.

    ValueError
        If 'x' array contains non numeric values.

    '''
    if not isinstance(m, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError("The coefficients 'm' and 'b' must be numeric")

    if not isinstance(x, (int, float, np.ndarray)):
        raise TypeError("The input variable 'x' must be an integer, float, or a numpy array")

    if isinstance(x, (int, float)):
        return m * x + b
    else:
        if not np.all(np.isreal(x)):
            raise ValueError("Array contains non-numeric values")
        ans_array = np.array([])
        for i in range(len(x)):
            ans_array = np.append(ans_array, m * x[i] + b)
        return ans_array
