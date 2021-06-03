class DeterministicWarmup:
    """Linear deterministic warm-up as described in [Sønderby 2016].

    Linearly decreases the temperature from t_start to t_max over then course of n iterations.
    If n == 0, the warmup is complete from the very first epoch, i.e. t=t_max (i.e. no warmup).
    """

    def __init__(self, n=200, t_max=1, t_start=0):
        if n < 0:
            raise ValueError("Cannot use fewer than zero iterations to warm up")
        self.n = n
        self.t_max = t_max
        self.t_start = t_start
        self.t = t_start if n != 0 else t_max  # If warming up over n=0 iterations, start at t_max, i.e. done.
        self.inc = 1 / n if n != 0 else 0  # If warmning up over n=0 iterations, incrememts are zero for completeness.

    @property
    def is_done(self):
        return self.t >= self.t_max

    def __iter__(self):
        return self

    def __next__(self):
        self.t += self.inc
        if self.is_done:
            return self.t_max
        return self.t

    def __repr__(self):
        s = f"DeterministicWarmup(n={self.n}, t_max={self.t_max}, t_start={self.t_start})"
        return s
