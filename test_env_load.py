import os
from pathlib import Path

# Test reading the .env file
env_path = Path(__file__).resolve().parent / ".env"
print(f"File path: {env_path}")
print(f"File exists: {env_path.exists()}")

if env_path.exists():
    try:
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    key = key.strip()
                    os.environ[key] = value.strip()
                    print(f"Set {key} = {value[:20]}...")
    except Exception as e:
        print(f"Error: {e}")

# Check if it was set
print(f"\nOPENAI_API_KEY in os.environ: {bool(os.getenv('OPENAI_API_KEY'))}")
print(f"OPENAI_API_KEY value: {os.getenv('OPENAI_API_KEY', 'NOT SET')[:30]}")
