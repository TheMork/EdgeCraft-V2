
import asyncio
import sys
import json
try:
    import websockets
except ImportError:
    print("websockets library not found. Please install it: pip install websockets")
    sys.exit(1)

async def test_ws():
    uri = "ws://127.0.0.1:8000/api/v1/simulation/ws?symbol=BTC/USDT&start_date=2024-01-01T00:00:00&end_date=2024-01-02T00:00:00&strategy=momentum"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    print(f"Received: {data.get('type')}")
                    if data.get('type') == 'status' and data.get('payload') == 'simulation_complete':
                        print("Simulation passed!")
                        break
                except websockets.exceptions.ConnectionClosed as e:
                    print(f"Connection closed: {e}")
                    break
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ws())
