try:
    from ._version import __version__
except ImportError:
    # Fallback when using the package in dev mode without installing
    # in editable mode with pip. It is highly recommended to install
    # the package from a stable release or in editable mode: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs
    import warnings

    warnings.warn("Importing 'data_ferret' outside a proper installation.")
    __version__ = "dev"

from jupyter_server.extension.application import ExtensionApp
from traitlets import Unicode
from data_ferret.util.output import timer


class DataFerretExtension(ExtensionApp):
    """Data Ferret server extension."""

    name = "data_ferret"
    load_other_extensions = True

    model = Unicode(
        default_value="gpt-4o",
        help="The model to use for the extension",
    ).tag(config=True)

    fast_model = Unicode(
        default_value="gpt-4o-mini",
        help="The fast model to use for the extension",
    ).tag(config=True)

    aliases = {
        "model": "DataFerretExtension.model",
        "fast-model": "DataFerretExtension.fast_model",
    }

    def initialize_settings(self):
        """Initialize settings for the extension."""
        with timer(message="Initializing Data Ferret settings..."):
            self.log.info(f"Initializing {self.name} extension")
            self.serverapp.web_app.settings["data_ferret"] = {
                "ext": self,
                "model": self.model,
                "fast-model": self.fast_model,
            }
        make_kernels()

    def initialize_handlers(self):
        """Register HTTP handlers for the extension."""
        with timer(message="Initializing Data Ferret handlers..."):
            from data_ferret.server.handlers import setup_handlers
            setup_handlers(self.serverapp.web_app)
            self.log.info(f"Registered {self.name} server extension handlers")


def _jupyter_labextension_paths():
    """Provide the location of the labextension."""
    with timer(message="JupyterLab extension paths..."):
        return [{"src": "labextension", "dest": "data_ferret"}]


def _jupyter_server_extension_points():
    """Define the server extension entry point."""
    with timer(message="Jupyter server extension points..."):
        return [{"module": "data_ferret", "app": DataFerretExtension}]


def make_kernels():
    from data_ferret.util.kernel_installer import install_kernel_spec
    from pathlib import Path

    install_kernel_spec(
        "ferret_kernel", Path(__file__).parent / "kernel" / "kernelspec"
    )
