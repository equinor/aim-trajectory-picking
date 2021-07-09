# aim-trajectory-picking
![GitHub](https://img.shields.io/github/license/Vildeeide/aim-trajectory-picking)
![GitHub contributors](https://img.shields.io/github/contributors/equinor/aim-trajectory-picking)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)


Trajectory picking package for the AI for Maturation project. Created by summer interns 2021

# Authors: ## 
Alexander Johnsgaard\
Even Åge Smedshaug\
Jonatan Lærdahl\
Vilde Dille Øvreeide\
Henrik Pettersen 

# Introduction
The goal of the project was to explore and discover algorithms that coordinate wellbore trajectories and pick paths that optimize different variables, given certain constraints.
Knowing from the start that the greedy algorithm performs well on the given problem despite its limitations, it became the baseline algorithm which the aim was to beat. To operationalize the trajectory picking problem, trajectories were considered nodes in a graph. Given this formulation, ideas and solutions from graph theory could be utilized to solve the problem. Each trajectory has a donor, a target and a non-negative value, in addition to a list of other trajectories it collides with. Given that no trajectories should share the same donor, nor target, and not collide with other trajectories, the task was to pick trajectories that maximize the sum of trajectory values. 

# NB Under heavy development ##




[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)

