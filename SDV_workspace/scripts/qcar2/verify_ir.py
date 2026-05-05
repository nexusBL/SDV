import cv2
import time
import numpy as np
from tessting import Supervisor

# Run supervisor in preview mode for a few seconds and capture a frame
try:
    app = Supervisor(preview_mode=True)
    app.camera.initialize()
    app.safety.initialize()
    app.depth.initialize()
    app.car.initialize()
    
    print("Stabilizing sensors...")
    for _ in range(30): # Warmup
        app.camera.get_frame()
        app.depth.get_obstacle()
        time.sleep(0.1)
        
    print("Capturing Verification Frame...")
    frame = app.camera.get_frame()
    error, hud = app.perception.process_frame(frame)
    
    # Manually trigger the state machine logic that populates HUD
    lidar_clear, lidar_dist = app.safety.is_path_clear()
    depth_reading = app.depth.get_obstacle()
    app.last_depth = depth_reading
    
    # Render HUD to a numpy array (imshow is handled by the script normally, 
    # but we want to save the result)
    app._render_hud(hud)
    # The _render_hud method calls cv2.imshow. We can just save 'hud' here 
    # because that's what _render_hud is drawing on.
    
    cv2.imwrite("verif_ir_hud.png", hud)
    print("✓ Verification frame saved to verif_ir_hud.png")
    
finally:
    try:
        app._shutdown()
    except:
        pass
