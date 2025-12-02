import os

# Read the .env file with UTF-16 encoding (to handle the BOM)
try:
    with open('.env', 'r', encoding='utf-16') as f:
        content = f.read()
    print("Successfully read as UTF-16")
    
    # Write it back as UTF-8
    with open('.env', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully converted to UTF-8")
    
    # Verify it can be read back
    with open('.env', 'r', encoding='utf-8') as f:
        verify = f.read()
    print(f"Verification successful. Content: {verify[:80]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
