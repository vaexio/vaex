# this uses https://pydoit.org/ to run tasks/chores
# pip install doit
# $ doit
import pkg_resources
import vaex.meta._version


def task_mybinder():
    """Make the mybinder files up to date"""

    def action(targets):
        filename = targets[0]
        with open(filename, "w") as f:
            version = vaex.meta._version.__version__
            version_normalized = pkg_resources.safe_version(version)
            f.write(f"vaex=={version_normalized}")
        print(f"{filename} updated")

    return {
        'actions': [action],
        'targets': ["binder/requirements.txt"],
        }
