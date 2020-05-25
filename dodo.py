# this uses https://pydoit.org/ to run tasks/chores
# pip install doit
# $ doit
import pkg_resources
import vaex.meta._version
import re
import shutil


def task_mybinder():
    """Make the mybinder files up to date"""

    def action(targets):
        filename = targets[0]
        with open(filename) as f:
            content = f.read()
        version = vaex.meta._version.__version__
        content = re.sub('vaex==(.*)', f'vaex=={version}', content)
        with open(filename, "w") as f:
            f.write(content)
        print(f"{filename} updated")

    return {
        'actions': [action],
        'targets': ["binder/requirements.txt"],
        'file_dep': ['packages/vaex-meta/vaex/meta/_version.py']
        }


def task_sync_readme():
    """Make the README for veax-meta up to date"""

    def action(targets):
        shutil.copy('README.md', targets[0])

    return {
        'actions': [action],
        'targets': ["packages/vaex-meta/README.md"],
        'file_dep': ['README.md']
        }
