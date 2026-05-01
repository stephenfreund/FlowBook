try:
    from ._version import __version__
except ImportError:
    # Fallback when using the package in dev mode without installing
    # in editable mode with pip. It is highly recommended to install
    # the package from a stable release or in editable mode: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs
    import warnings

    warnings.warn("Importing 'flowbook' outside a proper installation.")
    __version__ = "dev"

from jupyter_server.extension.application import ExtensionApp
from traitlets import Unicode
from flowbook.util.output import timer


class FlowBookExtension(ExtensionApp):
    """FlowBook server extension."""

    name = "flowbook"
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
        "model": "FlowBookExtension.model",
        "fast-model": "FlowBookExtension.fast_model",
    }

    def initialize_settings(self):
        """Initialize settings for the extension."""
        with timer(message="Initializing FlowBook settings..."):
            self.log.info(f"Initializing {self.name} extension")
            self.serverapp.web_app.settings["flowbook"] = {
                "ext": self,
                "model": self.model,
                "fast-model": self.fast_model,
            }
            km = self.serverapp.kernel_manager
            if km.default_kernel_name == "python3":
                km.default_kernel_name = "flowbook_kernel"
            ksm = self.serverapp.kernel_spec_manager
            if not ksm.allowed_kernelspecs:
                ksm.allowed_kernelspecs = {"flowbook_kernel", "python3"}
        make_kernels()

    def initialize_handlers(self):
        """Register HTTP handlers for the extension."""
        with timer(message="Initializing FlowBook handlers..."):
            from flowbook.server.handlers import setup_handlers
            setup_handlers(self.serverapp.web_app)
            self.log.info(f"Registered {self.name} server extension handlers")


def _jupyter_labextension_paths():
    """Provide the location of the labextension."""
    with timer(message="JupyterLab extension paths..."):
        return [{"src": "labextension", "dest": "flowbook"}]


def _jupyter_server_extension_points():
    """Define the server extension entry point."""
    with timer(message="Jupyter server extension points..."):
        return [{"module": "flowbook", "app": FlowBookExtension}]


def make_kernels():
    from flowbook.util.kernel_installer import install_kernel_spec
    from pathlib import Path

    base = Path(__file__).parent
    install_kernel_spec("flowbook_kernel", base / "kernel" / "kernelspec")
    install_kernel_spec("checkpoint_kernel", base / "checkpoint_kernel" / "kernelspec")
    install_kernel_spec("baseline_kernel", base / "baseline_kernel" / "kernelspec")
