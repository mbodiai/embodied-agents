import asyncio
import json
import logging

import websockets
from typer import Typer


async def ws_listen(self, timeout: int = 30) -> str:
    """Listens for audio input and transcribes it using a WebSocket endpoint.

    Args:
        timeout: The maximum time (in seconds) to wait for a transcription.

    Returns:
        The transcribed text from the audio input.
    """
    uri = "ws://3.22.171.235:5018/transcriptions"  # Updated endpoint
    auth_token = f"Bearer {self.client.api_key}" if not self.run_local else ""

    try:
        async with websockets.connect(uri, extra_headers={"Authorization": auth_token}) as websocket:
            # Send an initialization message if required by the server
            init_message = {"action": "start_transcription"}  # Adjust based on server protocol
            await websocket.send(json.dumps(init_message))
            logging.info("Sent initialization message to WebSocket.")

            # Stream the audio file in chunks wrapped in JSON with 'bytes' key
            with open(self.listen_filename, "rb") as audio_file:
                while True:
                    chunk = audio_file.read(1024)
                    if not chunk:
                        break
                    # Encode the binary data to base64 to include in JSON
                    import base64
                    encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                    message = {"bytes": encoded_chunk}
                    await websocket.send(json.dumps(message))
                    await asyncio.sleep(0.01)  # Small delay to prevent overwhelming the server

            # Signal end of audio stream if required
            end_message = {"action": "end_transcription"}  # Adjust based on server protocol
            await websocket.send(json.dumps(end_message))
            logging.info("Sent end message to WebSocket.")

            # Await transcription result
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                data = json.loads(response)
                if data.get("type") == "transcription":
                    transcription = data.get("data", {}).get("text", "")
                    logging.info(f"Received transcription: {transcription}")
                    return transcription
                logging.warning(f"Unexpected message type received: {data.get('type')}")
            except asyncio.TimeoutError:
                logging.error("Transcription timed out.")
    except websockets.exceptions.InvalidURI:
        logging.error(f"Invalid WebSocket URI: {uri}")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"WebSocket connection closed with error: {e}")
    except Exception as e:
        logging.error(f"An error occurred while connecting to WebSocket: {e}")

    return ""

def run_sync(func, *args, **kwargs):
    """Run an async function synchronously."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(func(*args, **kwargs))
if __name__ == "__main__":
    # Create a Typer app to run the WebSocket listener
    app = Typer()
    app.command()(run_sync(ws_listen))
    app()