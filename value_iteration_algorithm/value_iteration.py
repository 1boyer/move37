# This program implements the "value iteration" algorithm for determining a
# value landscape in a Markov Decision Process optimal control problem.

# The algorithm works by filling a grid with 'expected values' for optimal trajectories
# that move through those grid points. To fill in this grid we iterate a calculation
# over states and actions, filling in values as we go. The 'dynammic programming'
# insight is that we can calculate the expected value of the next state based on
# a prior state in memory (in an iterated fashion) rather than searching the space
# of trajectories which is exponentially large.
#
# Calculation the Value of States by Iteration:
# 1. Initialize lookup table of values with all zeros
# 2I. Loop over all states  
#   2II. Loop over all actions
#     2IIa. For each action get the list of tuples for (probability s->s', R(s',a),  s')
#           i.e. Given an action, what states s' are likely and what are the rewards
#     2IIb. Calculate expected reward
#     2IIc. Calculate the expected value of the next state
#     2IId. Calculate the expected value gain given an action [Hamiltonian Jacobi Bellman]
#     2IIe. Keep the action that leads to the highest expected value and that expected value
#  2III. Set state value to the highest expected value in 2IIe.
#  3. Iterate until now state value has changed substantially (arbitrary threshold)
#
#  Having calculated the value landscape, an optimal protocol can be determined
#  using a greedy algorithm very quickly.





from __future__ import print_function, division
from builtins import range
import sys
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
from grid_world import standard_grid
from utils import print_values, print_policy


# SMALL_ENOUGH is referred to by the mathematical symbol theta in equations
SMALL_ENOUGH = 1e-4
GAMMA = 0.9 # Discount Factor
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def optimal_action_value(grid, values, state):
# finds the highest value action (optimal_action) from state, returns the action and value
	grid.set_state(state)

	optimal_value = -100
	for action in ALL_POSSIBLE_ACTIONS:
		prob_reward_state_list = grid.get_transition_probs(action)
		expected_reward = 0 # Expectation value of reward given action
		expected_value = 0  # Expectation value of state given action
		for trans_prob, trans_reward, trans_state in prob_reward_state_list:
			expected_reward += trans_prob*trans_reward
			expected_value += trans_prob*values[trans_state]

		# Hamilton-Jacobi-Bellman equation
		# Value of current state _given an optimal decision is made_!
		expected_state_value = expected_reward + GAMMA*expected_value
		if expected_state_value > optimal_value:
			optimal_value = expected_state_value
			optimal_action = action

	return optimal_action, optimal_value

def calculate_values(grid):
	state_list = grid.all_states()

	# We must construct a lookup table for state:value
	# pairs. This is _THE KEY_ to an iterated solution
	# of the Hamilton-Jacobi-Bellman equation. 
	values = {} # Lookup table for States
	for state in state_list:
		values[state] = 0

	# Iterate until the value table isn't changing by more than SMALL_ENOUGH constant
	while True:
		value_delta = 0 # Variable to store the largest change in value this iteration
		for state in grid.non_terminal_states(): # check_action has an error if you feed in a terminal state
			optimal_action, state_value = optimal_action_value(grid, values, state)
			difference = np.abs(values[state]-state_value)
			values[state] = state_value
			if difference > value_delta:
				value_delta = difference
		if value_delta < SMALL_ENOUGH:
			break;

	return values


def calculate_greedy_policy(grid, values):
	policy = {}
	for state in grid.non_terminal_states():
		grid.set_state(state)
		optimal_action, _ = optimal_action_value(grid, values, state)
		policy[state] = optimal_action
	return policy


if __name__ == '__main__':	
	if len(sys.argv) > 2:
		try:
			obey_prob = float(sys.argv[1])
			step_cost = -float(sys.argv[2])
		except:
			print("Bad arguments: Usage python " + 
				sys.argv[0] + " obey_prob(float) + step_cost(float)")
			sys.exit()

	elif len(sys.argv) > 1:
		try:
			obey_prob = float(sys.argv[1])
			step_cost = 0
		except:
			print("Bad arguments: Usage python " + 
				sys.argv[0] + " obey_prob(float) + step_cost(float)")
			sys.exit()
	else:
		step_cost = 0
		obey_prob = 1.0
			

  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
	grid = standard_grid(obey_prob=obey_prob, step_cost=step_cost)

  # print rewards
	print("rewards:")
	print_values(grid.rewards, grid)

  # calculate accurate values for each square
	values = calculate_values(grid)

  # calculate the optimum policy based on our values
	policy = calculate_greedy_policy(grid, values)

  # our goal here is to verify that we get the same answer as with policy iteration
	print("values:")
	print_values(values, grid)
	print("optimal policy:")
	print_policy(policy, grid)
