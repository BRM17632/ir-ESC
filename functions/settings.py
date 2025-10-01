from pathlib import Path

def init():
    global project_name
    project_name = ""

    global current_wd
    current_wd = ""

def create_dir():
    Path(f"{current_wd}/Proyectos/{project_name}").mkdir(parents=True, exist_ok=True)
    Path(f"{current_wd}/Proyectos/{project_name}/Output").mkdir(parents=True, exist_ok=True)
    Path(f"{current_wd}/Proyectos/{project_name}/Progress").mkdir(parents=True, exist_ok=True)