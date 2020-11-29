import numpy as np
import itertools as it
import copy


class Comet:

    def __init__(self, criteria: list):
        """

        Parameters
        ----------
        criteria: list
            A list storing lists of criteria values
        """

        if not criteria:
            raise ValueError("Lack of criteria")

        if type(criteria) != list:
            raise TypeError("Wrong type of criteria: %s" % (repr(type(criteria))))

        for i, simpleCriterion in enumerate(criteria):
            for j, valueCriterion in enumerate(simpleCriterion):
                if type(valueCriterion) is not int and type(valueCriterion) is not float:
                    raise TypeError(
                        "Wrong type of %s value %s criterion: %s" % (repr(j), repr(i), repr(type(valueCriterion)))
                    )

        self._criteria: list = criteria
        self._sj: list = []
        self._co: np.ndarray = np.ndarray([0])

    def generate_co(self):
        """Generates characteristic object values based on the value of criteria"""

        criteria_iterator: list = list(it.product(*self._criteria))
        co: list = []

        for i, coord in enumerate(criteria_iterator):
            tmp: list = list(coord)
            tmp.append(0)
            co.append(tmp)

        self._co = np.asarray(co)

    def rate_co(self, preference: str = 'value', value: float = 0.0, mej: np.ndarray = np.ndarray([0])):
        """

        Parameters
        ----------
        preference: {'mej', 'random', 'value'}, optional
            Defines the value of preferences, default last when optional.
        value: int, optional
            Preference value that will be assigned to all characteristic objects (if preference is not 'mej' or
            'random'), default 0.0 value.
        mej: np.ndarray, optional
            MEJ matrix. Required when preference get 'mej' value.
        """

        if self._co.size == 0:
            raise ValueError(
                "Lack of characteristic objects"
            )

        if preference == 'mej':
            if mej.size == 0:
                raise ValueError(
                    "Lack of MEJ matrix"
                )
            else:
                p: list = self.__get_preference(mej)

        elif preference == 'random':
            p: list = list(np.random.rand(len(self._co)))

        elif preference == 'value':
            if value > 1 or value < 0:
                raise ValueError(
                    "Unexpected preference value: %s"
                    % (repr(value))
                )
            else:
                p: list = [value] * len(self._co)

        else:
            raise ValueError(
                "Unexpected preference option: %s"
                % (repr(preference))
            )

        self._co[:, -1] = p

    def change_co_preference(self, preference: list):
        """Changes the preference of characteristic objects

        Parameters
        ----------
        preference: list
            List of characteristic object preferences
        """
        if self._co.size == 0:
            raise ValueError(
                "Lack of characteristic objects"
            )

        if any(np.array(preference) > 1):
            raise ValueError(
                "Preference for a characteristic object too high"
            )

        if any(np.array(preference) < 0):
            raise ValueError(
                "Preference for a characteristic object too low"
            )

        self._co[:, -1] = preference

    def get_co_len(self):
        """Returns the number of characteristic objects

        Returns
        -------
            The number of characteristic objects
        """
        return len(self._co)

    def get_co_preference(self):
        """Returns preference values of characteristic objects

        Returns
        -------
            Preference values of characteristic objects
        """
        return self._co[:, -1]

    def rate(self, alternatives: np.ndarray) -> np.ndarray:
        """Calculates the value of preferences of given alternatives

        Parameters
        ----------
        alternatives: np.ndarray
            Numpy array of alternatives with their values for each criterion. Each alternative is a row in the array.

        Returns
        -------
            Numpy array of alternatives preferences
        """
        if self._co.size == 0:
            raise ValueError(
                "Lack of characteristic objects"
            )

        if not alternatives.shape[1] == self._co.shape[1] - 1:
            raise ValueError(
                "Incorrect number of criteria values for alternatives: %s" % (repr(alternatives.shape[0]))
            )

        for i in range(len(self._criteria)):
            if any(alternatives[:, i] < self._criteria[i][0]) or any(alternatives[:, i] > self._criteria[i][-1]):
                raise ValueError(
                    "The value of alternatives extends beyond bounds"
                )

        product: list = [[] for _ in range(len(alternatives))]
        preferences: np.ndarray = np.zeros(len(alternatives))
        alternatives: np.ndarray = copy.deepcopy(alternatives)

        for point in self._co:
            space_iter = self.__scope(point)

            for i, alternative in enumerate(alternatives):
                product[i] = [self.__tfn(coord, sp[0], sp[1], sp[2]) for sp, coord in zip(space_iter, alternative)]

            for i in range(len(alternatives)):
                preferences[i] += np.multiply(np.product(product[i]), point[-1])

            product = [[] for _ in range(len(alternatives))]

        return preferences

    def __get_preference(self, mej: np.ndarray) -> list:
        """Calculates preference for characteristic objects based on MEJ matrix"""

        self._sj: np.ndarray = np.zeros(len(mej))

        for i in range(len(self._sj)):
            self._sj[i] = sum(mej[i, :])

        sj_tmp: np.ndarray = copy.copy(self._sj)
        k: int = len(np.unique(self._sj))
        d: float = float(0)
        preference: np.ndarray = np.zeros(len(self._sj))

        for _ in range(k):
            index = np.where(sj_tmp == min(sj_tmp))
            preference[index] = float(d)
            d += 1 / (k - 1)
            sj_tmp[index] = max(sj_tmp) + 1

        return list(preference)

    def __scope(self, point: list) -> list:
        """Returns intervals in which the values of a characteristic object are contained

        Parameters
        ----------
        point: list
            Characteristic object

        Returns
        -------
            Characteristic space
        """
        scope_list: list = []
        for var, criterion in zip(point[:-1], self._criteria):
            ind: int = criterion.index(var)
            if ind == len(criterion) - 1:
                scope_list.append([criterion[ind - 1], criterion[ind], criterion[ind]])
            elif ind == 0:
                scope_list.append([criterion[ind], criterion[ind], criterion[ind + 1]])
            else:
                scope_list.append([criterion[ind - 1], criterion[ind], criterion[ind + 1]])

        return scope_list

    @staticmethod
    def __tfn(x: float, a: float, m: float, b: float) -> float:
        """Triangular Fuzzy Number method

        Parameters
        ----------
        x: float
            Representative crisp value
        a: float
            Smallest likely value
        m: float
            Most probable value
        b: float
            Largest possible value

        Returns
        -------
            Defuzzification value
        """
        if x < a or x > b:
            return float(0)

        elif a <= x < m:
            return (x-a) / (m-a)

        elif m < x <= b:
            return (b-x) / (b-m)

        elif x == m:
            return float(1)
