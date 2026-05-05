class ControlSystem:
    """
    PID Controller for steering the QCar 2 based on lane offset error.
    Includes anti-windup clamping and a low-pass filter on the derivative 
    term to smooth out steering jitter.
    """
    def __init__(self, kp=0.00225, ki=0.00015, kd=0.00075, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Target pixel offset from camera center. 
        # Non-zero because QCar 2 CSI camera 3 is physically mounted slightly off-center.
        self.setpoint = setpoint
        
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_d_term = 0.0
        
        # Anti-windup threshold to prevent integral runaway
        self.max_integral = 50.0  
        
        # Low pass filter coefficient for derivative term (0.0 = no filter, 0.9 = heavy filter)
        self.alpha_d = 0.3 

    def control_p(self, ubicacion_linea):
        """Simple proportional control"""
        error = self.setpoint + ubicacion_linea
        return error * self.kp
    
    def control_pid(self, ubicacion_linea):
        """Standard PID control step"""
        error = self.setpoint + ubicacion_linea
        
        # Proportional
        p_term = error * self.kp
        
        # Integral with Anti-windup clamping
        self.integral += error
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        i_term = self.ki * self.integral
        
        # Derivative with Low-pass filtering
        raw_d_term = self.kd * (error - self.previous_error)
        d_term = (self.alpha_d * self.last_d_term) + ((1.0 - self.alpha_d) * raw_d_term)
        
        # Save state for next iteration
        self.previous_error = error
        self.last_d_term = d_term
        
        # Output calculation
        return p_term + i_term + d_term

    @staticmethod
    def saturate(raw_steering, major=0.6, less=-0.6):
        """Clamps the steering value to physical limits"""
        return max(less, min(major, raw_steering))

    # Getters and Setters
    def set_kp(self, new_kp): self.kp = new_kp
    def get_kp(self): return self.kp
        
    def set_ki(self, new_ki): self.ki = new_ki
    def get_ki(self): return self.ki
        
    def set_kd(self, new_kd): self.kd = new_kd
    def get_kd(self): return self.kd
        
    def reset(self):
        """Resets dynamic memory states of the controller"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_d_term = 0.0
