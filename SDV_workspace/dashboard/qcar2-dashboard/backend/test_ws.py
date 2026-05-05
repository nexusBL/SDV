import asyncio
import websockets
import json

async def test_lidar_ws():
    uri = "ws://localhost:8000/ws/lidar"
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            for i in range(3):
                message = await websocket.recv()
                data = json.loads(message)
                print(f"[{i}] Received {len(data)} points")
                if len(data) > 0:
                    print(f"[{i}] Sample points: {data[:5]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_lidar_ws())
