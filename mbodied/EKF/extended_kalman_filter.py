import numpy as np

class ExtendedKalmanFilter:
    def __init__(
        self,
        state_dim,
        control_dim,
        observation_dim,
        initial_state=None,
        initial_covariance=None,
        process_noise_cov=None,
        measurement_noise_cov=None,
        uncertainty_percentage=0.1,
        is_linear_f=False,
        is_linear_h=False
    ):
        """
        For reference, see: https://gaoyichao.com/Xiaotu/resource/refs/PR.MIT.en.pdf chapter 3.3
        
        Initializes the EKF with dimensions for the state, control input, and observation vectors.

        Args:
        - state_dim (int): Dimension of the state vector.
        - control_dim (int): Dimension of the control vector.
        - observation_dim (int): Dimension of the observation vector.
        - initial_state (np.ndarray or None): Optional initial state vector.
        - initial_covariance (np.ndarray or None): Optional initial covariance matrix.
        - uncertainty_percentage (float): Percentage for initial state uncertainty.
        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.observation_dim = observation_dim
        self.is_linear_f = is_linear_f
        self.is_linear_h = is_linear_h

        if initial_state is not None:
            self.x = self.to_column_vector(initial_state)
        else:
            self.x = np.zeros((state_dim, 1))

        if initial_covariance is not None:
            self.P = initial_covariance
        else:
            if np.all(self.x == 0):
                self.P = np.eye(state_dim)
            else:
                variances = np.array([uncertainty_percentage * abs(value) for value in self.x.flatten()])
                self.P = np.diag(variances)

        self.Q = process_noise_cov if process_noise_cov is not None else np.eye(state_dim) * 0.5
        self.R = measurement_noise_cov if measurement_noise_cov is not None else np.eye(observation_dim) * 0.8

        self.F = np.eye(state_dim)
        self.B = np.zeros((state_dim, control_dim))
        self.H = np.eye(observation_dim, state_dim)

    def predict(self, u):
        """
        Prediction step of the EKF.

        Args:
        - u (np.array): Control input vector.
        """
        u = self.to_column_vector(u)
        self.x = self.f(self.x, u) # ¯μt = g(ut, μt−1)
       
        F_jacobian = self.jacobian_f(self.x, u) # Gt

        self.P = F_jacobian @ self.P @ F_jacobian.T + self.Q # ¯Σt = Gt Σt−1 GTt + Rt

    def update(self, z):
        """
        Update step of the EKF based on observation z.

        Args:
        - z (np.array): Observation vector.
        """
        z = self.to_column_vector(z)
        z_pred = self.h(self.x) # h(¯μt)

        H_jacobian = self.jacobian_h(self.x) # Ht

        y = z - z_pred # zt - h(¯μt)

        S = H_jacobian @ self.P @ H_jacobian.T + self.R # Ht ¯Σt HTt + Qt

        K = self.P @ H_jacobian.T @ np.linalg.inv(S) # Kt

        self.x = self.x + K @ y # μt = ¯μt + Kt(zt − h(¯μt))

        self.P = (np.eye(self.state_dim) - K @ H_jacobian) @ self.P # Σt = (I − Kt Ht) ¯Σt

    def f(self, x, u):
        """
        Non-linear state transition function (robot motion).

        Args:
        - x (np.array): Current state vector.
        - u (np.array): Control input vector.

        Returns:
        - (np.array): Predicted next state.
        """
        return self.F @ x + self.B @ u

    def jacobian_f(self, x, u, epsilon=1e-5):
        """
        Compute the Jacobian of the state transition function f with respect to the state x.

        Args:
        - x (np.array): Current state vector.
        - u (np.array): Control input vector.

        Returns:
        - (np.array): Jacobian matrix of f with respect to x.
        """
        if self.is_linear_f:
            return self.F

        state_dim = x.shape[0]
        F_jacobian = np.zeros((state_dim, state_dim))
        
        for i in range(state_dim):
            x_perturbed = np.copy(x)
            x_perturbed[i] += epsilon
            
            F_jacobian[:, i] = (self.f(x_perturbed, u) - self.f(x, u)).flatten() / epsilon

        return F_jacobian

    def h(self, x):
        """
        Non-linear observation function. To be defined by the specific system model.

        Args:
        - x (np.array): Current state vector.

        Returns:
        - (np.array): Predicted observation.
        """
        return self.H @ x

    def jacobian_h(self, x, epsilon=1e-5):
        """
        Compute the Jacobian of the observation function h with respect to the state x.

        Args:
        - x (np.array): Current state vector.

        Returns:
        - (np.array): Jacobian matrix of h with respect to x.
        """
        if self.is_linear_h:
            return self.H
        
        observation_dim = self.h(x).shape[0]
        state_dim = x.shape[0]
        H_jacobian = np.zeros((observation_dim, state_dim))
        
        for i in range(state_dim):
            x_perturbed = np.copy(x)
            x_perturbed[i] += epsilon
            
            H_jacobian[:, i] = (self.h(x_perturbed) - self.h(x)).flatten() / epsilon

        return H_jacobian

    def get_state(self):
        """
        Returns the current state estimate.
        
        Returns:
            np.ndarray: The current state estimate vector.
        """
        return self.x

    def get_covariance(self):
        """
        Returns the current covariance matrix.
        
        Returns:
            np.ndarray: The current state covariance matrix.
        """
        return self.P

    def get_measurement_prediction(self):
        """
        Computes the predicted measurement based on the current state estimate.
        
        Returns:
            np.ndarray: The predicted measurement vector.
        """
        return self.h(self.x)

    def to_column_vector(self, v):
        """
        Converts a vector to a column vector if it isn't already.

        Args:
        - v (np.array): Input vector.

        Returns:
        - (np.array): Column vector.
        # """
        return v.reshape(-1, 1) if v.ndim == 1 else v