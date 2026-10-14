from dataclasses import dataclass, fields


class BudgetExhausted(Exception):
    def __init__(self, counter: str) -> None:
        super().__init__(f"the {counter} budget is exhausted")
        self.counter = counter


@dataclass(frozen=True)
class Budget:
    """Amounts of the four metered resources of a run, used as a limit, a usage counter or a request."""

    model_requests: int = 0
    tool_calls: int = 0
    side_effects: int = 0
    output_bytes: int = 0

    @classmethod
    def counters(cls) -> tuple[str, ...]:
        return tuple(field.name for field in fields(cls))

    def shortfall(self, used: "Budget", need: "Budget") -> str | None:
        """Name the first counter this limit cannot cover for `used + need`, or None when everything fits."""
        for name in self.counters():
            if getattr(used, name) + getattr(need, name) > getattr(self, name):
                return name
        return None

    def minus(self, other: "Budget") -> "Budget":
        return Budget(**{name: getattr(self, name) - getattr(other, name) for name in self.counters()})
