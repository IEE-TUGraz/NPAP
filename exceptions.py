from typing import Dict, Any


class NPAPError(Exception):
    """Base exception for all Network Partitioning and Aggregation Package errors"""
    pass


class DataLoadingError(NPAPError):
    """Raised when data loading fails"""

    def __init__(self, message: str, strategy: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.strategy = strategy
        self.details = details or {}


class PartitioningError(NPAPError):
    """Raised when partitioning fails"""

    def __init__(self, message: str, strategy: str = None, graph_info: Dict[str, Any] = None):
        super().__init__(message)
        self.strategy = strategy
        self.graph_info = graph_info or {}


class AggregationError(NPAPError):
    """Raised when aggregation fails"""

    def __init__(self, message: str, strategy: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.strategy = strategy
        self.details = details or {}


class ElectricalCalculationError(AggregationError):
    """Raised when electrical calculations fail (Kron reduction, electrical distance, etc.)"""

    def __init__(self, message: str, calculation_type: str = None, details: Dict[str, Any] = None):
        super().__init__(message, calculation_type, details)
        self.calculation_type = calculation_type


class ValidationError(NPAPError):
    """Raised when validation fails"""

    def __init__(self, message: str, missing_attributes: Dict[str, list] = None,
                 strategy: str = None):
        super().__init__(message)
        self.missing_attributes = missing_attributes or {}
        self.strategy = strategy


class GraphCompatibilityError(NPAPError):
    """Raised when graphs are incompatible (e.g., partition doesn't match graph)"""

    def __init__(self, message: str, expected_hash: str = None, actual_hash: str = None):
        super().__init__(message)
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


class StrategyNotFoundError(NPAPError):
    """Raised when a requested strategy is not registered"""

    def __init__(self, strategy_name: str, strategy_type: str, available_strategies: list = None):
        message = f"Strategy '{strategy_name}' not found for {strategy_type}"
        if available_strategies:
            message += f". Available strategies: {', '.join(available_strategies)}"
        super().__init__(message)
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.available_strategies = available_strategies or []
