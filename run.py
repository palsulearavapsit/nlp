"""Launch The Review Room.

    python run.py            # serve on http://127.0.0.1:8000 and open a browser
    python run.py --port 9000 --no-browser

WHAT HAPPENS WHEN YOU RUN THIS
    1. uvicorn (a Python web server) starts and loads the FastAPI app defined
       in app/server.py (the object called ``api``).
    2. After 1.5 seconds a browser tab opens on the interface.
    3. The browser downloads web/index.html + web/assets/app.js, and app.js
       calls the /api/... endpoints as you click through the pages.
    Data files are loaded lazily, on the first request that needs them (see
    app/data.py), so start-up is instant.
"""

import argparse
import threading
import webbrowser

import uvicorn

# Plain ASCII: Windows consoles default to cp1252 and cannot encode box drawing.
BANNER = """
  ============================================================
    THE REVIEW ROOM
    NLP intelligence for the Amazon review corpus
  ============================================================
"""


def main():
    parser = argparse.ArgumentParser(description="Serve The Review Room.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window on start")
    parser.add_argument("--reload", action="store_true",
                        help="restart the server when source files change")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(BANNER)
    print(f"  Serving on {url}")
    print(f"  API docs   {url}/docs")
    print("  Press Ctrl+C to stop.\n")

    # Open the browser on a timer so the server has a moment to start first.
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    try:
        # "app.server:api" = the variable `api` inside app/server.py.
        uvicorn.run("app.server:api", host=args.host, port=args.port,
                    reload=args.reload, log_level="info")
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
