"""Provider-facing contract errors shared by loading boundaries."""


class ProviderContractError(ValueError):
    """A provider source violated an advertised or typed data contract."""
