
from pydantic import BaseModel, Field

class FerretConfig(BaseModel):
    """Configuration for Ferret commands.

    These defaults match the DataFerretExtension traitlet defaults,
    ensuring consistent behavior between Jupyter server and CLI usage.
    """
    model: str = Field(
        default="gpt-5.1",
        description="The model to use for AI-powered commands"
    )
    fast_model: str = Field(
        default="gpt-5.1-mini",
        description="The fast model to use for lightweight AI operations"
    )
