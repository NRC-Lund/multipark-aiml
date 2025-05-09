import os
from dotenv import load_dotenv

load_dotenv()

def get_env_var(name, required=True, default=None):
    value = os.getenv(name, default)
    if required and value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

def select_project(workspace, project_arg=None):
    """
    Select a project from the Roboflow workspace, either from argument or interactively.
    Returns the project slug (not workspace/project).
    """
    if project_arg:
        project_name = project_arg
    else:
        projects = workspace.projects()
        print("Available projects:")
        for idx, project in enumerate(projects):
            print(f"{idx+1}. {project}")
        while True:
            try:
                selection = int(input("Select a project by number: "))
                if 1 <= selection <= len(projects):
                    project_name = projects[selection-1]
                    break
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    print(f"Selected project string: {project_name}")
    if "/" in project_name:
        project_name = project_name.split("/")[-1]
    return project_name
