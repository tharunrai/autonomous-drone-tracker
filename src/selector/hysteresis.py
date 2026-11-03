"""Mode-switch hysteresis smoothing module to prevent frequent oscillation."""

from typing import Optional


class HysteresisSmoother:
    """Smoother for mode transitions with minimum duration and persistence thresholds."""

    def __init__(self, min_mode_duration: int = 15, persistence_frames: int = 5) -> None:
        """Initialize hysteresis smoother.

        Args:
            min_mode_duration: Minimum frames to stay in a mode before allowing a switch.
            persistence_frames: Consecutive frames candidate mode must be suggested to trigger switch.
        """
        self.min_mode_duration = min_mode_duration
        self.persistence_frames = persistence_frames
        self.current_mode: Optional[str] = None
        self.frames_in_current_mode: int = 0
        self.candidate_mode: Optional[str] = None
        self.candidate_count: int = 0

    def update(self, proposed_mode: str) -> str:
        """Update hysteresis state with the newly proposed mode and return filtered mode.

        Args:
            proposed_mode: The raw mode output predicted by the selector.

        Returns:
            The smoothed mode to be executed for the current frame.
        """
        raise NotImplementedError("HysteresisSmoother.update is a stub.")

    def reset(self, initial_mode: str = "FAST") -> None:
        """Reset internal counters to initial mode state.

        Args:
            initial_mode: Starting operational mode.
        """
        self.current_mode = initial_mode
        self.frames_in_current_mode = 0
        self.candidate_mode = None
        self.candidate_count = 0

