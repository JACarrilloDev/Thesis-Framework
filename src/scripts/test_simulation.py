from core.simulation import Simulation

def main():
    scene_file = "examples/warehouse_sorting/warehouse.blend"  # Replace with your CoppeliaSim scene file
    sim = Simulation(scene_file)
    try:
        sim.import_environment()
        sim.start()
        sim.load_robot()
        sim.train_ai(steps=10)  # Run a short training session
    finally:
        sim.shutdown()

if __name__ == "__main__":
    main()