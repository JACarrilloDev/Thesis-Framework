from pyrep import PyRep

def stop_simulation():
    pr = PyRep()
    pr.stop()
    pr.shutdown()
    print("Simulation stopped and shutdown successfully.")

if __name__ == "__main__":
    stop_simulation()