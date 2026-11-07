"""Structured rejection from a single Rust prepared-PDF parse invocation."""


class ParserResourceLimitError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        limit_name: str,
        limit: int,
        observed_or_requested: int,
        unit: str,
        scope: str,
        phase: str,
        page: int | None,
        object_ref: tuple[int, int] | None,
        policy_version: int,
    ) -> None:
        super().__init__(message)
        self.limit_name = limit_name
        self.limit = limit
        self.observed_or_requested = observed_or_requested
        self.unit = unit
        self.scope = scope
        self.phase = phase
        self.page = page
        self.object_ref = object_ref
        self.policy_version = policy_version

    def __reduce__(self):
        return (_restore_parser_resource_error, (self.args[0], self.__dict__))


def _restore_parser_resource_error(message, attributes):
    fields = (
        "limit_name",
        "limit",
        "observed_or_requested",
        "unit",
        "scope",
        "phase",
        "page",
        "object_ref",
        "policy_version",
    )
    error = ParserResourceLimitError(
        message, **{name: attributes[name] for name in fields}
    )
    error.__dict__.update(attributes)
    return error
