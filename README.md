# aim-trajectory-picking
![g_diggers](https://img.shields.io/badge/gold-diggers-yellow)
![GitHub](https://img.shields.io/github/license/Vildeeide/aim-trajectory-picking)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![GitHub contributors](https://img.shields.io/github/contributors/equinor/aim-trajectory-picking)

<em>Trajectory picking package for the AI for Maturation project. Created by summer interns 2021</em>

##  Description
This trajectory picking package is a tool for determining sets of optimal wellbore trajectories, using different algorithms.
The trajectory picking task is operationalized as a graph problem, where each trajectory has a donor, a target and a non-negative value, in addition to a list of trajectories it collides with. Given that no trajectories should share the same donor, nor target, and not collide with other trajectories, different algorithms are implemented which attempt to pick optimal trajectories. The goal of the project was to explore the problem space and implement an algorithm with better performance than the greedy algorithm.


## Getting started 
```
pip install git+https://github.com/equinor/aim-trajectory-picking.git@master
pip install -r test_requirements.txt
pip install .
pytest
```
This package is built using NetworkX and Python 3.

## How to use the package
The package can be run from the command line. The input file should contain a list of trajectory dictionaries, which is examplified below:  

```
{"trajectories": [
        {"id": 0,
         "donor": "D0",
         "target": "T0",
         "value": 10,
         "collisions": [1, 2, 3, 4, 5]},
         
         {...]}}
```
### Console script commands 

| Command        | Action                                                |
|----------------|-------------------------------------------------------|
| run            | --                                                    |
| pytest         | Running all available tests                           |
| benchmark      | Showing pre-generated  figure of expected performance |

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)
## Known bugs:
- ValueError: x and y must have same first dimension, but have shapes (8,) and (14,) \
Solution: delete results.txt. Results will have to be recalculated, but at leas the plotting works.
### Authors: ## 
Alexander Johnsgaard\
Even Åge Smedshaug\
Jonatan Lærdahl\
Vilde Dille Øvreeide\
Henrik Pettersen 

