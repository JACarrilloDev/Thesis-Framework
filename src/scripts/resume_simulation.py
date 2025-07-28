from pyrep import PyRep

def resume_simulation():
    pr = PyRep()
    pr.start()
    print("Simulation resumed.")

if __name__ == "__main__":
    resume_simulation()