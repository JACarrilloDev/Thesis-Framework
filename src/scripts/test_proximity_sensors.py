from pyrep import PyRep
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.backend import sim
import time

SCENE_FILE = "user_workspace/custom_scenes/minimal_sensor_test.ttt"

pr = None
try:
    pr = PyRep()
    pr.launch(SCENE_FILE, headless=False)
    pr.start()

    print("Stepping simulation to initialize...")
    for _ in range(5):
        pr.step()
    time.sleep(0.5)

    print("\nAttempting to read sensor...")
    sensor = ProximitySensor("Proximity_sensor")
    
    # Try both APIs
    print("Testing PyRep API...")
    raw_result = sensor.read()
    print(f"PyRep result: {raw_result}")
    
    print("\nTesting direct sim API...")
    handle = sim.simGetObjectHandle("Proximity_sensor")
    # Changed: Only unpack what the API actually returns
    result = sim.simReadProximitySensor(handle)
    print(f"Direct sim API result: {result}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    if pr is not None and pr.running:
        pr.stop()
        pr.shutdown()