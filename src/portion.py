import itertools
import numpy as np


class Interval:
    repr_round = 2

    def __init__(self, *args):
        borders = []

        if (
            len(args) == 2
            and isinstance(args[0], (float, int))
            and isinstance(args[1], (float, int))
        ):
            if args[0] <= args[1]:
                borders = [(args[0], args[1])]
        else:
            for a in args:
                assert isinstance(a, (tuple, list)) and len(a) == 2
                if a[0] <= a[1]:
                    borders.append((a[0], a[1]))

        self.borders = self._squeeze(borders)

    def empty(self):
        return not self.borders

    def lower(self):
        if not self.borders:
            return np.nan
        return min(x for x, _ in self.borders)

    def upper(self):
        if not self.borders:
            return np.nan
        return max(x for _, x in self.borders)

    def length(self):
        return sum(e - s for s, e in self.borders)

    def get_closest_value(self, v):
        if not self.borders:
            return np.nan

        if v in self:
            return v

        value, distance = np.nan, np.inf
        for start_end in self.borders:
            for b in start_end:
                if abs(b - v) < distance:
                    value, distance = b, abs(b - v)

        return value

    def __print_value(self, v):
        if self.repr_round is None:
            return v
        return round(v, self.repr_round)

    def __repr_borders(self, start, end):
        if start == end:
            return f"[{self.__print_value(start)}]"
        else:
            return f"[{self.__print_value(start)}, {self.__print_value(end)}]"

    def __repr__(self):
        if not self.borders:
            return "[]"
        return " | ".join([self.__repr_borders(*b) for b in self.borders])

    def __bool__(self):
        return bool(self.borders)

    def __eq__(self, other):
        if len(self.borders) != len(other.borders):
            return False
        return all(b1 == b2 for b1, b2 in zip(self.borders, other.borders))

    def __contains__(self, x):
        for s, e in self.borders:
            if s <= x <= e:
                return True
        return False

    def __and__(self, other):
        borders = []
        for s1, e1 in self.borders:
            for s2, e2 in other.borders:
                borders.append((max(s1, s2), min(e1, e2)))
        return Interval(*borders)

    def __or__(self, other):
        borders = self.borders + other.borders
        return Interval(*borders)

    def __neg__(self):
        if self.empty():
            return Interval(-np.inf, np.inf)

        out = Interval()

        if self.lower() > -np.inf:
            out |= Interval(-np.inf, self.lower())

        if self.upper() < np.inf:
            out |= Interval(self.upper(), np.inf)

        for (s1, e1), (s2, e2) in zip(self.borders[:-1], self.borders[1:]):
            out |= Interval(e1, s2)

        return out

    def __sub__(self, other):
        out = self & -other
        new_borders = []
        for s, e in out.borders:
            if s == e and s in other:
                continue
            new_borders.append((s, e))
        out.borders = new_borders
        return out

    def __add__(self, v: float):
        return Interval(*[(s + v, e + v) for s, e in self.borders])

    @staticmethod
    def _squeeze(borders):
        if len(borders) < 2:
            return borders

        def maybe_intersect(s1, e1, s2, e2):
            if s2 <= s1 <= e2:
                return s2, max(e1, e2)
            if s2 <= e1 <= e2:
                return min(s1, s2), e2
            if s1 <= s2 and e1 >= e2:
                return s1, e1

        flag = True
        while flag:
            flag = False
            for i, j in itertools.combinations(range(len(borders)), r=2):
                if i == j or not borders[i] or not borders[j]:
                    continue

                r = maybe_intersect(*borders[i], *borders[j])
                if r:
                    borders[i], borders[j] = r, None
                    flag = True

        borders = [x for x in borders if x]
        return sorted(borders, key=lambda x: x[0])
