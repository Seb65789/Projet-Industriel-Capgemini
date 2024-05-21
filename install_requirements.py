import subprocess

# commande pour l'installation
install = "pip install -r requirements.txt"

run = subprocess.run(install,capture_output=True,shell=True)


print(run.stderr)