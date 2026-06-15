import os
import stat

def configure_cli(target_dir='.', templates_dir=None):
    container = os.environ['SINGULARITY_CONTAINER'] # error out early
    name = os.path.splitext(os.path.basename(container))[0]
    script_name = f'run_{name}_sif.sh'
    script = os.path.join(target_dir,script_name)

    from McUtils.ExternalPrograms import SLURMClient
    SLURMClient.create_server_package(target_dir, overwrite=True)

    if templates_dir is None:
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    with open(os.path.join(templates_dir, 'container_runner.sh'), 'r') as f:
        src = f.read()
        src = src.replace('`CONTAINER_PATH`', container)
        with open(script, 'w+') as dest:
            dest.write(src)

    sbatch_name = f'{name}_sif_python.sh'
    sbatch_path = os.path.join(target_dir, sbatch_name)
    with open(os.path.join(templates_dir, 'container_sbatch.sh'), 'r') as f:
        src = f.read()
        src = src.replace('`CONTAINER_SCRIPT_NAME`', script_name)
        with open(sbatch_path, 'w+') as dest:
            dest.write(src)

    st = os.stat(script)
    os.chmod(script, st.st_mode | stat.S_IEXEC)

    return script

if __name__ == "__main__":
    import sys
    configure_cli(*sys.argv[1:])