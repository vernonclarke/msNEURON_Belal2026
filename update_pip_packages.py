import pkg_resources
from subprocess import call

# List all installed packages
packages = [dist.project_name for dist in pkg_resources.working_set]

# Update each package
call('pip install --upgrade ' + ' '.join(packages), shell=True)
