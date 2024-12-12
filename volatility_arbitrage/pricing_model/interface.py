"""interface of pricing model"""

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

ARRAY = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HestonParams:
    """Heston stochastic volatilty model parameters"""

    kappa: float
    mean_of_var: float
    vol_of_var: float
    rho: float

    def __post_init__(self) -> None:
        assert self.mean_of_var > 0
        assert self.kappa > 0
        assert self.vol_of_var > 0
        assert abs(self.rho) < 1
        # Check feller condition
        assert 2 * self.kappa * self.mean_of_var > self.vol_of_var**2


@dataclass(frozen=True)
class MarketModel:
    """Market model"""

    imp_model: HestonParams
    real_model: HestonParams
    rho_spot_imp_var: float
    rho_real_var_imp_var: float

    def __post_init__(self) -> None:
        assert abs(self.rho_spot_imp_var) <= 1
        assert abs(self.rho_real_var_imp_var) <= 1


@dataclass
class StrategyPnl:
    """strategy pnl class"""

    total_pnl: ARRAY
    var_vega_pnl: ARRAY
    theta_pnl: ARRAY
    vanna_pnl: ARRAY
    gamma_pnl: ARRAY
    vega_hedge_pnl: ARRAY

    def __add__(self, other: "StrategyPnl") -> "StrategyPnl":
        """Dynamically add two StrategyPnl objects using asdict()."""
        if not isinstance(other, StrategyPnl):
            raise TypeError(f"Cannot add {type(other)} to StrategyPnl")
        self_dict = asdict(self)
        other_dict = asdict(other)

        result = {key: value + other_dict[key] for key, value in self_dict.items()}
        return StrategyPnl(**result)

    def __sub__(self, other: "StrategyPnl") -> "StrategyPnl":
        """Dynamically subtract two StrategyPnl objects using asdict()."""
        if not isinstance(other, StrategyPnl):
            raise TypeError(f"Cannot sub {type(other)} to StrategyPnl")
        self_dict = asdict(self)
        other_dict = asdict(other)

        result = {key: value - other_dict[key] for key, value in self_dict.items()}
        return StrategyPnl(**result)


@dataclass(frozen=True)
class StrategyPnlCalculator:
    """strategy pnl calculator"""

    total_pnl: ARRAY
    var_vega_pnl: ARRAY
    theta_pnl: ARRAY
    vanna_pnl: ARRAY
    gamma_pnl: ARRAY
    vega_hedge_pnl: ARRAY

    def get_strategy_pnl(self, position: npt.NDArray[np.float64]) -> StrategyPnl:
        """calculate pnl of a strategy"""
        return StrategyPnl(
            total_pnl=self.total_pnl * position,
            var_vega_pnl=self.var_vega_pnl * position,
            theta_pnl=self.theta_pnl * position,
            vanna_pnl=self.vanna_pnl * position,
            gamma_pnl=self.gamma_pnl * position,
            vega_hedge_pnl=self.vega_hedge_pnl * position,
        )
