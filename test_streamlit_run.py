
import subprocess
import time
import sys

print("Starting Streamlit app...")
process = subprocess.Popen(['./venv/bin/streamlit', 'run', 'streamlit_app.py', '--server.headless=true'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

time.sleep(5)

if process.poll() is None:
    print("Streamlit app is running successfully.")
    process.terminate()
    sys.exit(0)
else:
    stdout, stderr = process.communicate()
    print("Streamlit app failed to start.")
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    sys.exit(1)
