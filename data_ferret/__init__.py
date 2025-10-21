try:
    from ._version import __version__
except ImportError:
    # Fallback when using the package in dev mode without installing
    # in editable mode with pip. It is highly recommended to install
    # the package from a stable release or in editable mode: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs
    import warnings

    warnings.warn("Importing 'data_ferret' outside a proper installation.")
    __version__ = "dev"
from data_ferret.util.output import timer
from data_ferret.server.handlers import setup_handlers


def _jupyter_labextension_paths():
    with timer(message="JupyterLab extension paths..."):
        return [{"src": "labextension", "dest": "data_ferret"}]


def _jupyter_server_extension_points():
    with timer(message="Jupyter server extension points..."):
        return [{"module": "data_ferret"}]


def _load_jupyter_server_extension(server_app):
    """Registers the API handler to receive HTTP requests from the frontend extension.

    Parameters
    ----------
    server_app: jupyterlab.labapp.LabApp
        JupyterLab application instance
    """
    with timer(message="Loading Jupyter server extension..."):
        setup_handlers(server_app.web_app)
        name = "data_ferret"
        server_app.log.info(f"Registered {name} server extension")


def make_kernels():
    from data_ferret.util.kernel_installer import install_kernel_spec
    from pathlib import Path

    install_kernel_spec(
        "ferret_kernel", Path(__file__).parent / "kernel" / "kernelspec"
    )


make_kernels()
