from .Comet import Comet
import numpy as np
import copy
from mlrose import hill_climb, simulated_annealing, ContinuousOpt, CustomFitness
from pyswarms import single


# 	Stochastic IdenTifiCation Of Model
class Sitcom:

    def __init__(self, alternatives: np.ndarray, preference: list, criteria: list, stochastic_method: str = 'pso',
                 iterations: int = 1000):
        """

        Parameters
        ----------
        alternatives: np.ndarray
            Numpy array of alternatives with their values for each criterion. Each alternative is a row in the array.
        stochastic_method: {'hill-climbing', 'simulated-annealing', 'pso'}
            Defines a stochastic optimization method
        criteria: list
            A list storing lists of criteria values
        iterations: int, optional
            Number of iterations in the optimization method
        """

        for i in range(len(criteria)):
            if any(alternatives[:, i] < criteria[i][0]) or any(alternatives[:, i] > criteria[i][-1]):
                raise ValueError(
                    "The value of alternatives extends beyond bounds"
                )

        if any(np.array(preference) > 1):
            raise ValueError(
                "Preference for an alternatives too high"
            )

        if any(np.array(preference) < 0):
            raise ValueError(
                "Preference for an alternatives too low"
            )

        for i, simpleCriterion in enumerate(criteria):
            for j, valueCriterion in enumerate(simpleCriterion):
                if type(valueCriterion) is not int and type(valueCriterion) is not float:
                    raise TypeError(
                        "Wrong type of %s value %s criterion: %s" % (repr(j), repr(i), repr(type(valueCriterion)))
                    )

        self._alternatives: np.ndarray = alternatives
        self._alternativesPreference: list = preference
        self._stochasticMethod: str = stochastic_method
        self._criteria: list = criteria
        self._iterations: int = iterations

    def generate_criteria(self):
        """Creates criteria based on a alternatives"""

        if self._criteria:
            self._criteria = []

        n, _ = self._alternatives.shape

        for number in range(n):
            self._criteria.append([np.min(self._alternatives[:, number]), np.mean(self._alternatives[:, number]),
                                   np.max(self._alternatives[:, number])])

    def rate(self, co_type: str, co_value: float):
        """Calculates the preference of characteristic objects using the defined stochastic method.

        Parameters
        ----------
        co_type: str
            Type of preference values of objects characteristic of the COMET method
        co_value: float
            The value of preferences of objects characteristic of the COMET method
        Returns
        -------

        """

        model = Comet(self._criteria)
        model.generate_co()
        model.rate_co(co_type, co_value)

        dict_arg = {'model': model, 'alternatives': self._alternatives, 'preference': self._alternativesPreference}

        if self._stochasticMethod == "hill-climbing":
            problem = ContinuousOpt(model.get_co_len(), CustomFitness(self._mlrose_fitness, **dict_arg),
                                    maximize = False, step = 0.01)
            pos, _, cost_history = hill_climb(problem, max_iters = self._iterations, curve = True,
                                              init_state = model.get_co_preference())
            cost_history = np.abs(cost_history)

        elif self._stochasticMethod == "simulated-annealing":
            problem = ContinuousOpt(model.get_co_len(), CustomFitness(self._mlrose_fitness, **dict_arg),
                                    maximize = False, step = 0.01)
            pos, _, cost_history = simulated_annealing(problem, max_iters = self._iterations, curve = True,
                                                       init_state = model.get_co_preference())
            cost_history = np.abs(cost_history)

        elif self._stochasticMethod == "pso":
            bound_max = np.ones(model.get_co_len())
            bound_min = np.zeros(model.get_co_len())
            bounds = (bound_min, bound_max)
            options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

            optimizer = single.GlobalBestPSO(n_particles = 20, dimensions = model.get_co_len(), options = options,
                                             bounds = bounds)
            cost, pos = optimizer.optimize(self._pso_fitness, self._iterations, **dict_arg)
            cost_history = optimizer.cost_history

        else:
            raise ValueError(
                "Wrong optimization method has been determined: %s"
                % (repr(self._stochasticMethod))
            )

        return pos, cost_history

    @staticmethod
    def _pso_fitness(x: np.ndarray, model: Comet, alternatives: np.ndarray, preference: list):
        """Fitness function for the PSO method

        Parameters
        ----------
        x: np.ndarray
            Position of the particles
        model: Comet
            Comet model with preference for characteristic objects
        alternatives: np.ndarray
            Numpy array of alternatives with their values for each criterion. Each alternative is a row in the array.
        preference:
            Reference preference for alternatives
        Returns
        -------
            Fitness function values for particles
        """

        fit_values = []
        for pref in x:
            model_tmp = copy.deepcopy(model)
            model_tmp.change_co_preference(pref)
            preference_tmp = model_tmp.rate(alternatives)
            fit_values.append(np.sum(np.abs(np.subtract(preference_tmp, preference))))
        return fit_values

    @staticmethod
    def _mlrose_fitness(x: np.ndarray, model: Comet, alternatives: np.ndarray, preference: np.ndarray):
        """Fitness function for the hill climbing and simulated annealing methods

        Parameters
        ----------
        x: np.ndarray
            Numpy array of preference values for characteristic objects
        model: Comet
            Comet model with preference for characteristic objects
        alternatives: np.ndarray
            Numpy array of alternatives with their values for each criterion. Each alternative is a row in the array.
        preference:
            Reference preference for alternatives
        Returns
        -------
            Fitness function value
        """

        model_tmp = copy.deepcopy(model)
        model_tmp.change_co_preference(list(x))
        preference_tmp = model_tmp.rate(alternatives)
        return np.sum(np.abs(np.subtract(preference_tmp, preference)))
