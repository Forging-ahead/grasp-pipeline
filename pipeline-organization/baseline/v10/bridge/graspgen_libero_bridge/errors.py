"""Failures describe pipeline stages, not unobserved grasp outcomes."""


class ExecutionError(RuntimeError):
    def __init__(self, message, *, code="execution_error", category="execution",
                 phase=None, details=None):
        super().__init__(message)
        self.code, self.category, self.phase = code, category, phase
        self.details = {} if details is None else details

    def record(self):
        return dict(code=self.code, category=self.category, phase=self.phase,
                    details=self.details)
