import subprocess
import sys
import time
import webbrowser
import os


# Get the folder where run.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dashboard_path = os.path.join(BASE_DIR, "dashboard.py")

print("=" * 60)
print("   AMAZON REVIEW NLP INTELLIGENCE SYSTEM")
print("=" * 60)
print()
print("Starting dashboard...")
print()


# Start Streamlit
process = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        dashboard_path,
        "--server.headless",
        "true"
    ],
    cwd=BASE_DIR
)


# Give Streamlit time to start
time.sleep(5)


# Open dashboard in browser
url = "http://localhost:8501"

print(f"Opening dashboard: {url}")
webbrowser.open(url)

print()
print("Dashboard is running.")
print("Keep this terminal open while using the dashboard.")
print("Press Ctrl+C to stop the project.")
print()


try:
    process.wait()

except KeyboardInterrupt:

    print()
    print("Stopping dashboard...")

    process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()

    print("Dashboard stopped.")