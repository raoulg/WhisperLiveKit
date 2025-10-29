from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import (
    TranscriptionEngine,
    AudioProcessor,
    get_inline_ui_html,
    parse_args,
)
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the TranscriptionEngine when the server starts."""
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    """Serves the main HTML user interface."""
    return HTMLResponse(get_inline_ui_html())


async def handle_websocket_results(websocket, results_generator, output_file=None):
    """Consumes results from the audio processor, sends them to the client, and saves them to a file."""
    last_written_content = ""
    try:
        async for response in results_generator:
            response_dict = response.to_dict()
            await websocket.send_json(response_dict)

            if output_file and "lines" in response_dict:
                formatted_lines = []
                for line in response_dict["lines"]:
                    speaker = line.get("speaker", "UNKNOWN")
                    start = line.get("start", "0:00:00")
                    end = line.get("end", "0:00:00")
                    text = line.get("text", "").strip()
                    if text:
                        formatted_lines.append(
                            f"[{start} -> {end}] Speaker {speaker}: {text}"
                        )

                current_content = "\n".join(formatted_lines)

                if current_content != last_written_content:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(current_content + "\n")
                    last_written_content = current_content

        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected while handling results (client likely closed connection)."
        )
    except Exception as e:
        logger.exception(f"Error in WebSocket results handler: {e}")


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """Handles the WebSocket connection for real-time transcription."""
    global transcription_engine
    # Safely get the output_file argument, defaulting to None if not provided
    output_file = getattr(args, "output_file", None)

    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    try:
        # Send initial configuration to the client
        use_audio_worklet = getattr(args, "pcm_input", False)
        await websocket.send_json(
            {"type": "config", "useAudioWorklet": bool(use_audio_worklet)}
        )
    except Exception as e:
        logger.warning(f"Failed to send config to client: {e}")

    results_generator = await audio_processor.create_tasks()
    # Create a task to handle sending results back to the client and saving them
    websocket_task = asyncio.create_task(
        handle_websocket_results(websocket, results_generator, output_file)
    )

    try:
        # Loop to receive audio data from the client
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(
            f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True
        )
    finally:
        # Clean up tasks and resources on disconnect
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")

        await audio_processor.cleanup()
        logger.info("WebSocket endpoint cleaned up successfully.")


def main():
    """Entry point for the CLI command to run the Uvicorn server."""
    import uvicorn

    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    # Safely check for SSL arguments before using them
    if hasattr(args, "ssl_certfile") and hasattr(args, "ssl_keyfile"):
        if args.ssl_certfile or args.ssl_keyfile:
            if not (args.ssl_certfile and args.ssl_keyfile):
                raise ValueError(
                    "Both --ssl-certfile and --ssl-keyfile must be specified together."
                )
            ssl_kwargs = {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }

    if ssl_kwargs:
        uvicorn_kwargs.update(ssl_kwargs)

    # Safely check for forwarded_allow_ips argument before using it
    if hasattr(args, "forwarded_allow_ips") and args.forwarded_allow_ips:
        uvicorn_kwargs["forwarded_allow_ips"] = args.forwarded_allow_ips

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
