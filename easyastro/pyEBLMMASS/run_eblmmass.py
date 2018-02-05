import subprocess


def run_eblmmass():
	sp = subprocess.Popen(["/bin/bash", "-i", "-c", "star; eblmmass < pyEBLMMASS_input.txt"])
	sp.communicate()
