
import sys
import os
import inspect
sys.path.append(os.getcwd())
try:
    from questdb.ingress import Sender
    print("Sender imported successfully.")
    print("Sender signature:")
    print(inspect.signature(Sender.__init__))
    print("Sender class docstring:")
    print(Sender.__doc__)
except ImportError:
    print("Could not import Sender.")
except Exception as e:
    print(f"Error inspecting Sender: {e}")
