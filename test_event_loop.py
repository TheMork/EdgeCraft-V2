
import sys
import os
sys.path.append(os.getcwd())

def test_event_loop():
    print("Attempting to import simulation_core...")
    try:
        import simulation_core
        print("simulation_core imported successfully.")
        print(f"simulation_core contents: {dir(simulation_core)}")
    except ImportError as e:
        print(f"Failed to import simulation_core: {e}")
        return

    print("Attempting to instantiate EventLoop...")
    try:
        from src.simulation.event_loop import EventLoop
        loop = EventLoop(latency_ms=10)
        print("EventLoop instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate EventLoop: {e}")

if __name__ == "__main__":
    test_event_loop()
