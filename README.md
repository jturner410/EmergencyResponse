# EmergencyResponse
A physics-based program that simulates a swarm of agents travelling through randomly generated obstacles toward a final goal

The number of agents and obstacles can be changed by updating the variables in the main function. Obstacles are of random size and strength, and will change position through each running of the simulation.

The agents will avoid the obstacles and choose the most direct path to the goal. If the agents don't make it to the goal, the simulation regards it as a fail.

The main file runs the simulation 50 times, also a changable amount, and aggregates the percentage of agents that make it to the goal. A final heatmap is printed to show the distribution of the obstacles over the course of every simulation. The separate visualize file is included for a step-by-step view of the swarm's navigation through the obstacle course (of the most recent run).
