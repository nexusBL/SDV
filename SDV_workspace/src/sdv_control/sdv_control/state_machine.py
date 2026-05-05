#!/usr/bin/env python3
"""
Adaptive Cruise Control (ACC) State Machine
Translates Perception Threat Levels & Lane states into safe throttle actuation logic.
"""
import time
from sdv_perception.config import SDVConfig

class StateMachine:
    """Evaluates perception inputs and enforces driving safety rules."""
    
    STATE_IDLE = 'IDLE'                 # Car disabled/waiting
    STATE_DRIVING = 'DRIVING'           # Fully Autonomous (Threat = CLEAR)
    STATE_AVOIDING = 'AVOIDING'         # Slowed down (Threat = WARNING)
    STATE_EMERGENCY_STOP = 'EMERGENCY_STOP' # Hard Brake (Threat = CRITICAL)
    STATE_LOST_LANE = 'LOST_LANE'       # Lost lane sight, coasting

    def __init__(self, config=None):
        self.cfg = config or SDVConfig.get()
        self.speed_cfg = self.cfg.speed
        
        self.current_state = self.STATE_IDLE
        self.active_throttle = 0.0

    def evaluate(self, threat_level: str, lane_status: str) -> float:
        """
        Evaluate current perception state and return safe throttle output.
        threat_level: "CLEAR", "WARNING", "CRITICAL", "UNKNOWN"
        lane_status: "BOTH", "LEFT_ONLY", "RIGHT_ONLY", "NONE"
        """
        # Highest Priority: Obstacle directly ahead
        if threat_level == 'CRITICAL':
            self.current_state = self.STATE_EMERGENCY_STOP
            self.active_throttle = self.speed_cfg['min_throttle'] # applying braking force
            return self.active_throttle
            
        # High Priority: Lost track of where the lane is
        if lane_status == "NONE":
            self.current_state = self.STATE_LOST_LANE
            self.active_throttle = 0.0  # coast to stop safely
            return self.active_throttle
            
        # Medium Priority: Obstacle in warning zone
        if threat_level == 'WARNING':
            self.current_state = self.STATE_AVOIDING
            self.active_throttle = self.speed_cfg['base_throttle'] * self.speed_cfg['warning_throttle_multiplier']
            return self.active_throttle
            
        # Default: Pure autonomy navigation
        self.current_state = self.STATE_DRIVING
        self.active_throttle = self.speed_cfg['base_throttle']
        return self.active_throttle
