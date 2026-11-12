"""Failures retain their stage and native backend details."""


class EpisodeFailure(RuntimeError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class EpisodeStopped(EpisodeFailure):
    """No further physical action may be executed in this episode."""
