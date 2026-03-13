import subprocess


result = subprocess.run(
    ["echo", "hello from the shell"],
    capture_output = True,
    text = True)

if result.returncode == 0:
    output = result.stdout.strip()
    print(f"The Agent captured: {output}")
else:
    print(f"Error occurred {result.stderr}")