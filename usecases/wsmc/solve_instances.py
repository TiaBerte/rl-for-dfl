"""
    Utility methods to solve and evaluate the Set Cover instances.
"""

import numpy as np
from ortools.linear_solver import pywraplp


########################################################################################################################


def compute_cost(instance, decision_vars, not_satisfied_demands):
    """
    Compute the true cost of a solution for the MSC.
    :param instance: usecases.setcover.generate_instances.MinSetCover; the problem instance.
    :param decision_vars: numpy.array of shape (num_sets, ); the solution.
    :param not_satisfied_demands: numpy.array of shape (num_prods, ); product demands that were not satisfied.
    :return: float; the true cost.
    """
    # Compute cost for using the sets
    real_cost = instance.set_costs @ decision_vars

    # Compute the cost for not satisfied demands
    not_satisfied_demands = np.clip(not_satisfied_demands, a_min=0, a_max=None)
    not_satisfied_demands_cost = not_satisfied_demands @ instance.prod_costs

    cost = real_cost + not_satisfied_demands_cost

    return cost


########################################################################################################################

class MinSetCoverProblem:
    """
    Class for the Minimum Set Cover problem.
    """

    def __init__(self, instance, output_flag=0):
        # Create the model
        self._solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self._solver:
            raise Exception("Failed to create the solver")

        infinity = self._solver.infinity()

        self._decision_vars = dict()

        for j in range(instance.num_sets):
            self._decision_vars[j] = self._solver.IntVar(0, infinity, 'x[%i]' % j)

        for i in range(instance.num_products):
            constr_expr = [instance.availability[i][j] * self._decision_vars[j] for j in range(instance.num_sets)]
            self._solver.Add(sum(constr_expr) >= instance.demands[i])

        obj_expr = [instance.set_costs[j] * self._decision_vars[j] for j in range(instance.num_sets)]
        self._solver.Minimize(self._solver.Sum(obj_expr))

    def solve(self):
        """
        Solve the optimization problem.
        :return: numpy.array, float; solution and its objective value.
        """
        status = self._solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            obj_val = self._solver.Objective().Value()
            solution = [self._decision_vars[j].solution_value() for j in range(len(self._decision_vars))]
        else:
            raise Exception("Problem has not a optimal solutions")

        print_str = ""
        for idx in range(len(solution)):
            print_str += f'Set n.{idx}: {solution[idx]} - '
        print_str += f'\nSolution cost: {obj_val}'

        return solution, obj_val
