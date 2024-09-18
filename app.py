# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# def main():
#     st.title("Polynomial Function Fitting App")

#     # Get the degree of the polynomial
#     degree = st.sidebar.slider("Select the degree of the polynomial", 1, 10, 2)

#     # Input number of points
#     num_points = st.sidebar.number_input("Number of points", min_value=2, value=5)

#     st.write("## Input Coordinates")
    
#     # Input coordinates
#     x_coords = []
#     y_coords = []
#     for i in range(num_points):
#         x = st.number_input(f"x{i+1}", value=float(i))
#         y = st.number_input(f"y{i+1}", value=float(i**2))
#         x_coords.append(x)
#         y_coords.append(y)

#     # Fit a polynomial to the input points
#     coefficients = np.polyfit(x_coords, y_coords, degree)
#     polynomial = np.poly1d(coefficients)

#     st.write("## Polynomial Coefficients")
#     st.write(coefficients)

#     # Input an x value for prediction
#     x_value = st.number_input("Input a value of x to predict y", value=0.0)
#     y_value = polynomial(x_value)
#     st.write(f"Predicted y value for x = {x_value} is {y_value}")

#     # Plotting
#     st.write("## Graph of the Function")
#     x_range = np.linspace(min(x_coords) - 1, max(x_coords) + 1, 100)
#     y_range = polynomial(x_range)

#     plt.figure(figsize=(10, 5))
#     plt.scatter(x_coords, y_coords, color='red', label='Input Points')
#     plt.plot(x_range, y_range, label='Fitted Polynomial', color='blue')
#     plt.scatter(x_value, y_value, color='green', label='Predicted Point')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Polynomial Fit')
#     plt.legend()
#     st.pyplot(plt)

# if __name__ == '__main__':
#     main()

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline

def main():
    st.title("Function Fitting and Interpolation App")

    # Dropdown to select the type of function/interpolation method
    method = st.sidebar.selectbox(
        "Select the type of function/interpolation",
        ("Polynomial Fitting", "Linear Interpolation", "Cubic Spline Interpolation")
    )

    # Get the degree of the polynomial (only relevant for Polynomial Fitting)
    degree = 2
    if method == "Polynomial Fitting":
        degree = st.sidebar.slider("Select the degree of the polynomial", 1, 10, 2)

    # Input number of points
    num_points = st.sidebar.number_input("Number of points", min_value=2, value=5)

    st.write("## Input Coordinates")
    
    # Input coordinates
    x_coords = []
    y_coords = []
    for i in range(num_points):
        x = st.number_input(f"x{i+1}", value=float(i))
        y = st.number_input(f"y{i+1}", value=float(i**2))
        x_coords.append(x)
        y_coords.append(y)

    # Interpolation or Polynomial Fitting
    if method == "Polynomial Fitting":
        coefficients = np.polyfit(x_coords, y_coords, degree)
        polynomial = np.poly1d(coefficients)

        # Define the function for predictions
        def predict(x):
            return polynomial(x)

        st.write("## Polynomial Coefficients")
        st.write(coefficients)

    elif method == "Linear Interpolation":
        linear_interp = interp1d(x_coords, y_coords, kind='linear', fill_value="extrapolate")

        # Define the function for predictions
        def predict(x):
            return linear_interp(x)

    elif method == "Cubic Spline Interpolation":
        spline_interp = make_interp_spline(x_coords, y_coords, k=3)

        # Define the function for predictions
        def predict(x):
            return spline_interp(x)



    # Plotting
    st.write("## Graph of the Function")
    x_range = np.linspace(min(x_coords) - 1, max(x_coords) + 1, 100)
    y_range = predict(x_range)

    plt.figure(figsize=(10, 5))
    plt.scatter(x_coords, y_coords, color='red', label='Input Points')
    plt.plot(x_range, y_range, label=method, color='blue')
    # plt.scatter(x_value, y_value, color='green', label='Predicted Point')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Fitting and Interpolation')
    plt.legend()
    st.pyplot(plt)
    
    # Input an x value for prediction
    x_value = st.number_input("Input a value of x to predict y", value=0.0)
    y_value = predict(x_value)
    st.write(f"Predicted y value for x = {x_value} is {y_value}")

if __name__ == '__main__':
    main()
