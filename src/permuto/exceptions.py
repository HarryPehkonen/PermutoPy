# src/permuto/exceptions.py

class PermutoException(Exception):
    """Base class for all Permuto exceptions."""
    pass

class PermutoInvalidOptionsError(PermutoException, ValueError):
    """Exception raised for invalid configuration options."""
    pass

class PermutoParseException(PermutoException):
    """Exception raised for errors parsing template/context (less common)."""
    pass

class PermutoCycleError(PermutoException):
    """Exception raised when a cyclical dependency is detected."""
    def __init__(self, message: str, cycle_path_info: str = ""):
        super().__init__(message)
        self.cycle_path_info = cycle_path_info

    def __str__(self) -> str:
        base = super().__str__()
        if self.cycle_path_info:
            return f"{base} Cycle detected involving: [{self.cycle_path_info}]"
        return base

class PermutoMissingKeyError(PermutoException, LookupError):
    """Exception raised when a key/path is not found and error mode is set."""
    def __init__(self, message: str, key_path: str = ""):
        super().__init__(message)
        self.key_path = key_path

    def __str__(self) -> str:
        base = super().__str__()
        if self.key_path:
            return f"{base} Path: [{self.key_path}]"
        return base

class PermutoReverseError(PermutoException, RuntimeError):
    """Exception raised during reverse template operations."""
    pass
