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
As shown below, the id's start at 0 and incrementally increase by 1, which is crucial for the program's functions using indexing.

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
| run            | Runs the program and gives outputs in text and graph  |
| pytest         | Running all available tests (read "testing")          |
| benchmark      | Showing pre-generated  figure of expected performance |

#### Command specifications:
| run            | Explanation                                                                          |
|----------------|--------------------------------------------------------------------------------------|
| -alg           | specify algorithms to run. "all" runs all algorithms                                 |
| -datasets      | specify datasets to run algorithm on. full path of folder with datasets is required  |
| -refresh True  | Ignores if the result have already been calculated and does a new calculation        |

#### Examples:
run -alg all -datasets full_path_of_datasetsfolder  
Explanation: Runs all the algorithms on specified datasets

run -alg greedy -datasets full_path_of_datasetsfolder  
Explanation: Runs greedy algorithm on specified datasets

run -alg greedy weight_trans lonely_target -datasets random 15 15 5000 0.05 5  
Explanation: Runs greedy-, weight_transformation- and lonely_target algorithm on 5 randomly generated datasets with 15 donors, 15 targets, 5000 trajectories, 5% collision rate.

[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)

### Testing

| Command                               | Action                                        |
|---------------------------------------|-----------------------------------------------|
| pip install -r test_requirements.txt  | Installs the required packages to run tests   |
| pytest                                | Runs all available tests                      |

## Useful links
networkx: https://networkx.org/
OR-Tools: https://developers.google.com/optimization
JSON: https://docs.python.org/3/library/json.html
argparse: https://docs.python.org/3/library/argparse.html
igraph: https://igraph.org/python/doc/tutorial/tutorial.html

## Further ideas
In the end of the internship we had some ideas we unfortunately did not have time to implement.

Firstly, it would have been interesting to make a heuristic which cluster trajectories in space. Here a suggestion is that every donor and target could contain information about where it is located in space by, for instance, x-, y- and z-coordinates. Then the computer would know where each trajectory starts (donor coordinates) and ends (target coordinates). If the donor and target associated with one trajectory is far away from the donor and target associated with another trajectory, the idea is that those two trajectories are not likely to collide in space. In other words, if one implements a clustering algorithm which clusters trajectories which are relatively close to each other in space, then our algorithms can be run on each of these clusters. This would be under the assumption that trajectories in different clusters don't collide with each other, since they would have different donors and targets, adn the donors and targets are far away from each other, so that collisions in space also are highly unlikely. Since all of the clusters now can be considered as independent sets of trajectories, the algorithms could be run on each of these smaller sets, resulting in less computations and lower runtimes.      

Secondly, it would be smart to try out another package from python than networkx, as it seems to take much time. We tried to use igraph instead of networkx, and this needed only minor changes in the code for networkx. The result was lower runtimes. However, the results when using igraphs was actually worse than the ones using networkx, and despite the improvements in runtime, we therefore decided not to go on with igraph. However, the improvements show that it could be interesting to try out outher graphical tools than networkx.

## Alternatives (OR-Tools)

An alternative to the algorithms we have developed is to solve this problem by integer-linear-programming. This could be done by usng OR-Tools which we have implemented in this project. This worked better than our algorithms, but on the other hand, the runtime was higher. However, for small problems with sufficiently few trajectories, we recommend ILP to be used.

## Known bugs:
- None

### Authors: ## 
Alexander Johnsgaard\
Even Åge Smedshaug\
Jonatan Lærdahl\
Vilde Dille Øvreeide\
Henrik Pettersen 

