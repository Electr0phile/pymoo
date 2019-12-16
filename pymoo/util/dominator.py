import numpy as np


class Dominator:

    @staticmethod
    def get_relation(a, b, ax, bx):

        # returns -1  if a is dominated by b
        #         1 if b is dominated by a
        #         0  if a and b have no domination relationship

        a_infeas = ax[0].get_infeasability()
        b_infeas = bx[0].get_infeasability()

        if a_infeas > b_infeas:
            # a is less feasable, b dominates a
            return -1
        elif a_infeas < b_infeas:
            # b is less feasable, a dominates b
            return 1

        val = 0
        for i in range(len(a)):
            if a[i] < b[i]:
                # indifferent because once better and once worse
                if val == -1:
                    return 0
                val = 1
            elif b[i] < a[i]:
                # indifferent because once better and once worse
                if val == 1:
                    return 0
                val = -1
        return val

    @staticmethod
    def calc_domination_matrix_loop(X, F):
        n = F.shape[0]
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                M[i, j] = Dominator.get_relation(F[i, :], F[j, :], X[i], X[j])
                M[j, i] = -M[i, j]

        return M

    @staticmethod
    def calc_domination_matrix(X, F, _F=None, epsilon=0.0):

        """
        if G is None or len(G) == 0:
            constr = np.zeros((F.shape[0], F.shape[0]))
        else:
            # consider the constraint violation
            # CV = Problem.calc_constraint_violation(G)
            # constr = (CV < CV) * 1 + (CV > CV) * -1

            CV = Problem.calc_constraint_violation(G)[:, 0]
            constr = (CV[:, None] < CV) * 1 + (CV[:, None] > CV) * -1
        """

        if _F is None:
            _F = F

        # look at the obj for dom
        n = F.shape[0]
        m = _F.shape[0]

        L = np.repeat(F, m, axis=0)
        R = np.tile(_F, (n, 1))

        smaller = np.reshape(np.any(L + epsilon < R, axis=1), (n, m))
        larger = np.reshape(np.any(L > R + epsilon, axis=1), (n, m))

        M = np.logical_and(smaller, np.logical_not(larger)) * 1 \
            + np.logical_and(larger, np.logical_not(smaller)) * -1

        # if cv equal then look at dom
        # M = constr + (constr == 0) * dom

        return M
