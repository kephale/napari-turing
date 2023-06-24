from ._TuringPattern import TuringPattern, ModelParameter
from scipy.ndimage import convolve
import numpy as np
from typing import Optional

class Oregonator(TuringPattern):
    default_size = 200
    default_dx = 1
    default_dy = 1
    default_dt = 0.01
    default_contrast_limits = (0.3, 3.5)

    A = ModelParameter(
        name="A",
        value=1.0,
        min=0.1,
        max=5.0,
        exponent=1,
        description="Concentration of productor of X",
    )
    B = ModelParameter(
        name="B",
        value=1.0,
        min=0.1,
        max=5.0,
        exponent=1,
        description="Concentration of productor of Y (combined with X)",
    )
    C = ModelParameter(
        name="C",
        value=0.1,
        min=0.1,
        max=5.0,
        exponent=1,
        description="Concentration of inhibitor",
    )
    mu_x = ModelParameter(
        name="mu_x",
        value=2.0e-5,
        min=0.1e-5,
        max=5.0e-5,
        exponent=1,
        description="Diffusion coefficient of X",
    )
    mu_y = ModelParameter(
        name="mu_y",
        value=1.0e-5,
        min=0.01e-5,
        max=20.0e-5,
        exponent=0.1,
        description="Diffusion coefficient of Y",
    )
    mu_z = ModelParameter(
        name="mu_z",
        value=1.0e-5,
        min=0.01e-5,
        max=20.0e-5,
        exponent=0.1,
        description="Diffusion coefficient of Z",
    )
    nb_pos = ModelParameter(
        name="nb_pos",
        value=1,
        min=1,
        max=300,
        exponent=1,
        description="Number of random perturbations",
        dtype=int
    )
    _necessary_parameters = [A, B, C, mu_x, mu_y, mu_z, nb_pos]
    _tunable_parameters = [A, B, C, mu_x, mu_y, mu_z, nb_pos]
    _concentration_names = ["X", "Y", "Z"]

    def _reaction(self, c: str) -> np.ndarray:
        if c == "X":
            return self.A - self.X + self.X**2 * self.Y
        elif c == "Y":
            return self.B * (self.X - self.X**2 * self.Y - self.C * self.Y)
        elif c == "Z":
            return self.C * (self.X - self.Z)

    def _diffusion(self, c: str) -> np.ndarray:
        if c == "X":
            arr = self.X
            mu = self.mu_x
        elif c == "Y":
            arr = self.Y
            mu = self.mu_y
        elif c == "Z":
            arr = self.Z
            mu = self.mu_z
        to_cell = convolve(arr, self.kernel.value, mode="constant", cval=0)
        from_cell = self.nb_neighbs * arr
        out = mu * (to_cell - from_cell) / (self.dx * self.dy)
        return out

    def init_concentrations(self, C: Optional[str] = None) -> None:
        pos = (np.random.random((2, self.nb_pos)) * self.size).astype(int)
        values = np.random.random(self.nb_pos)
        if C == "X" or C is None:
            X = np.ones((self.size, self.size)) * self.A
            X[pos[0], pos[1]] += values
            self["X"] = X
        if C == "Y" or C is None:
            Y = np.ones((self.size, self.size)) * (self.B / self.A)
            Y[pos[0], pos[1]] -= values
            self["Y"] = Y        
        if C == "Z" or C is None:
            Z = np.ones((self.size, self.size)) * (self.B / self.A)
            Z[pos[0], pos[1]] -= values
            self["Z"] = Z
