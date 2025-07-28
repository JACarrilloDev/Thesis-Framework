from pyrep import PyRep

def pause_simulation():
    pr = PyRep()
    pr.stop()
    print("Simulation paused.")

if __name__ == "__main__":
    pause_simulation()