# aim-trajectory-picking
![g_diggers](https://img.shields.io/badge/gold-diggers-yellow)
![GitHub](https://img.shields.io/github/license/Vildeeide/aim-trajectory-picking)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![GitHub contributors](https://img.shields.io/github/contributors/equinor/aim-trajectory-picking)

<em>Trajectory picking package for the AI for Maturation project. Created by summer interns 2021</em>.

###  Description
This trajectory picking package is a tool for determining sets of optimal wellbore trajectories, using different algorithms.
For our own algorithm implementations the problem has been treated as a graph problem, where each trajectory is a node and has a donor, a target and a non-negative value, in addition to a list of trajectories it collides with. These collisions are therefor treated as the edges in the graph between trajectories. Given that no trajectories should share the same donor, nor target, and not collide with other trajectories, different algorithms are implemented which attempt to pick optimal trajectories. The goal of the project was to explore the problem space and implement an algorithm which outperforms the greedy algorithm. Although weight transformation was shown to be a better algorithm compared to greedy, its runtime is problematic. Google's OR-Tools performs better and faster, and that is why it has been set as the default solver for this program. A seperate section about the ORTools implementation can be found further down.


### Getting started 
```
pip install git+https://github.com/equinor/aim-trajectory-picking.git@master
pip install -r test_requirements.txt
pip install .
pytest
```
This package is built using Python 3.

### How to use the package
The package can be run from the command line. The input file should contain list of trajectory dictionaries, which is examplified below:  
```
[
        {
         "id": 0,
         "donor": "D0",
         "target": "T0",
         "value": 10,
         "collisions": [1, 2, 3, 4, 5]},
         
         {...}]
```
The program finds the optimal trajectories and saves them to a .json file as a dictionary, where the keys are dataset name(s) and the values are a list of ID's 
which correspond to the optimal trajectories. Example:
```
{
        "dataset_1.json": [
                id_1,
                id_2,
                ...,
                id_n
                ],
        "dataset_2.json":[...]
                ...
        "dataset_k.json":[...]
}
```

#### Console script commands 

| Command        | Action                                                |
|----------------|-------------------------------------------------------|
| run            | Runs the program and gives outputs as text and graph  |
| pytest         | Running all available tests (read "testing")          |

#### Command specifications:

| run            | Explanation                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------|
| -alg           | Specify algorithms to run. "all" runs all algorithms                                               |
| -datasets      | Specify datasets to run algorithm on. Full path of the dataset folder  is required                 |
| --refresh / -r | Takes no arguments. Ignores if the result has already been calculated and does a new calculation   |
| -outputfile    | Specifies outputfile to write results to. Default is 'optimal_trajectories.json'                   |
| --verbose / -v | Takes no arguments. If given, prints more information in the process of calculating trajectories.  |

 
#### Examples:

| Command | Explanation |
|---|---|
| run  | Shows pre-generated figure of expected performance when no dataset is available, based on randomly pre-generated datasets with 100-5000 trajectories. |
| run -alg all -datasets full_path_of_datasetsfolder | Runs all the algorithms on specified datasets. |
| run -alg greedy -datasets full_path_of_datasetsfolder -outputfile local_file_name | Runs greedy algorithm on specified datasets. |
| run -alg greedy weight_trans lonely_target -datasets random 15 15 5000 0.05 5 | Runs greedy-, weight_transformation- and lonely_target algorithm on 5 randomly generated datasets with 15 donors, 15 targets, 5000 trajectories, 5% collision rate. |

### Testing

| Command                               | Action                                        |
|---------------------------------------|-----------------------------------------------|
| pip install -r test_requirements.txt  | Installs the required packages to run tests   |
| pytest                                | Runs all available tests                      |

### OR-Tools

An alternative to using the algorithms we have developed is to solve this problem as a optimization problem. This has been done using Google Optimization Research Tools (aka OR-Tools), which we have implemented in this project. By formulating the optimization problem where the objective function is the minimum weighted vertex cover (WVC), contraints could be added to account for collisions between trajectories and shared donor/target pairs. Since the minimum WVC contains the nodes not in the maximum weighted independent set (WIS), one only needs to solve either one to find the other. It should be noted this does not necessary hold for approximations to the exact solutions, however this is not a concern for our case.

Two methods from OR-Tools have been tested, Mixed Integer Programming (MIP) where the variables are allowed to be arbitrary integers, and an Assignment Problem with contraints. The MIP was initially tested since the exact vertex cover could be set up as such a problem. This worked better than our algorithms, but as the problem is NP-complete, the runtime was higher. Next, since the task of assigning the trajectories to either do or do not, the problem could instead be treated as a Constraint Programming problem, using a Satisfiability method (CP-SAT). This time the variables would either be assigned a 0 or 1 and only one trajectory from each list of collisions could be included in the maximum WIS. This proved to be incredibly effective and solved the exact solution faster than the approximations we had ourself implemented. Seemingly, the runtime is linear with number of trajectories but exponential with number of constraints. Since the number rof contraints is far less than the number of trajectories, this did not make a huge impact on the total runtime

### Useful links
- networkx: https://networkx.org/ 
- OR-Tools: https://developers.google.com/optimization
- JSON: https://docs.python.org/3/library/json.html
- argparse: https://docs.python.org/3/library/argparse.html
- igraph: https://igraph.org/python/doc/tutorial/tutorial.html


### Authors: ## 
Alexander Johnsgaard\
Even Åge Smedshaug\
Jonatan Lærdahl\
Vilde Dille Øvreeide\
Henrik Pettersen 


[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/Naereen/)
