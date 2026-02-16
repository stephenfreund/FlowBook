
from pydantic import BaseModel, Field

class FlowbookConfig(BaseModel):
    """Configuration for FlowBook commands.

    These defaults match the FlowBookExtension traitlet defaults,
    ensuring consistent behavior between Jupyter server and CLI usage.
    """
    model: str = Field(
        default="claude-opus-4-5",
        description="The model to use for AI-powered commands"
    )
    fast_model: str = Field(
        default="claude-opus-4-5",
        description="The fast model to use for lightweight AI operations"
    )
