# Equinor Summer Intern AIM Project 2021

## 1 Introduction
This report includes discoveries, reflections and results for the AIM trajectory picking project, carried out by five summer interns. The goal of the project was to explore and discover algorithms that coordinate wellbore trajectories and pick paths that optimize different variables, given certain constraints.
Knowing from the start that the greedy algorithm performs well on the given problem despite its limitations, it became the baseline algorithm which the aim was to beat. 

## 2 Setting up the problem
To operationalize the trajectory picking problem, trajectories were initially considered nodes in a graph. Given this formulation, ideas and solutions from graph theory could be utilized to solve the problem. Each trajectory has a donor, a target and a non-negative value, in addition to a list of other trajectories it collides with. Given that no trajectories should share the same donor, nor target, and not collide with other trajectories, the task was to pick trajectories that maximize the sum of trajectory values. 

## 3 Algorithms
During the working process we developed multiple modifications to the greedy algorithm, as well as new heuristic algorithms. 

### 3.1 Greedy

#### 3.1.1 The Original Greedy
With the goal of finding an algorithm which outperforms greedy, it was useful to start off by implementing greedy so it could be used as a baseline for comparisons. Here the Python package "NetworkX" was used to create a graph where each node represents a trajectory. The edges between these nodes imply that the corresponding trajectories are mutually exclusive, meaning that the trajectories either have the same donor, same target, or collide with each other. The algorithm was implemented so it first picks the node of highets value, then removes the neighbors of this node which now cannot be picked, and then chooses the node of highest value of the remaining nodes. This is done iteratively until there are no nodes left to pick, and the total value is the sum of the values of all picked nodes.

#### 3.1.2 Greedy Considering Value Blocked
One of the main weaknesses of the greedy algorithm described above is that it only considers picking nodes of high value. However, to maximize the total value of trajectories picked, we do not only want nodes of high value, but also many nodes. For the greedy algorithm to take both of these two factors into account, the greedy algorithm also considering values that were blocked was implemented. This algorithm is very similar to the greedy first implemented; the only difference is that instead of picking the node of highest value at all times, this algorithm instead picks the node with the highest value, after dividing it by the sum of the neighbours' value that the chosen node blocks.  

#### 3.1.3 Greedy Considering Number of Nodes Blocked
Another alternative to the initial greedy is an algorithm which was also motivated by the idea of considering blocked values. However, instead of using the sum of blocked values in the denominator, the denominator in this algorithm equals the number of nodes blocked, plus 1. The extra "1" is added to avoid a possible zero in the denominator. In comparison to the greedy considering blocked values, this function focuses on picking nodes that block few other nodes, rather than on picking nodes that block little value 

#### 3.1.4 Reversed Greedy
All of the three greedy variants explained above are strategies with the intention of picking valuable nodes from a set of nodes. The reversed greedy, however, focuses on eliminating nodes which block many other nodes. This algorithm uses the greedy algorithm to eliminate these nodes. When some of the nodes that block high values are removed, there are three ways in which the rest of the problem can be solved. The first and second methods are by using the original greedy or the greedy considering value blocked. The third way is to solve for the remaining collisions by using the greedy algorithm, and then solve for donors and targets by bipartite matching.

#### 3.1.5 Greedy Focusing on Lonely Targets
The last version of the greedy algorithm was built on the idea that when many donors are hit, the probability of getting a high total value is increased, even though not all of the trajectories are chosen based on value. This algorithm firstly focuses on choosing all the targets only hit by one trajectory and then uses the greedy algortihm. In this way we choose all the targets that remove the least amount of other trajectories before starting the regular greedy algorithm.

### 3.2 Random Choice
This algortihm, "Random Choice", randomly chooses its nodes in every step of the greedy algorithm, not considering their value, nor blockages. It was implemented to get an intuition of how good the greedy algorithm whi,ch picks nodes of high value, actually is. The purpose of this is that our heuristics now can be compared to the random algorithm, and not only the greedy algorithm we already know works quite well.

### 3.3 Bipartite Matching
The bipartite matching algorithm uses bipartite matching to pick trajectories. In this case a graph where each node represents a trajectory is still used, but with edges only for collisions; they do not take collision in donors or targets into account. Here the greedy algorithm is used to pick a set of nodes which will only contain non-colliding trajectories. Now that the collisions are taken care of, the challenge of donors and targets is solved by making a bipartite graph where the two independent sets of nodes are donors and targets, respectively, and the edges between them illustrate the trajectories. Therefore the edges are weighted by value. A built-in function in networkx is then used to perform the bipartite matching on this set of nodes, and the resulting list of nodes are the ones picked by this strategy.

### 3.4 Bipartite Matching, version 2
This bipartite matching algorithm is very similar to the one described above. The only difference is that, instead of solving for collisions by using the greedy algorithm, it uses the reversed greedy algorithm.

### 3.5 Reversed Bipartite Matching
This algorithm is similar to the Bipartite Matching algorithm, but works in the opposite direction. This function first solves for donors and targets by using bipartite matching, and solves for collisions afterwards. If any of the trajectories chosen from the bipartite matching collides, then the ones of lowest value are removed. Then, the algorithm looks for the trajectories of highest value which can replace the removed trajectories.

## 4 Exact solutions

### 4.1 Maximum (Weighted) Independent Set
The problem of picking the best trajectories can be considered a maximum weighted independent set problem, and is a so called strictly NP-hard problem. Since there are no solution to such a problem in polynomial time, it would be unrealistic to use this algorithm for a full sized dataset. Instead, testing has been done to determine how large a dataset one could reasonably use such an algorithm on. Other approaches could be to solve the minimum weighted vertex cover and take the trajectories not in the vertex cover or the maximum weighted clique problem on the complimentary graph. These are also so called NP-hard problems, which means the runtime may not see much improvement when solved exact. The maximum clique algorithm from Networkx found the exact result of the initial test sets generated. However due to the exponential runtime of this exact solution, even just for datasets of as few as 200 trajectories, the time taken to solve the maximum clique became unrealisically high. Therefore approximations were deemed necessary.

### 4.2 Approximations to the MIS algorithm
Approximations of the afore mentioned problems may result in exact or near-exact results at a fraction of the runtime that the exact method would use. An approximation to the minimum weighted vertex cover was tried, but the results were sub-par with out other algorithms we had implemented already. Additionally, the approximation would not result in a similar goodness approximation of the maximum weighted independent set since the latter lacks a constant approximation factor.

## 5 OR-Tools
An alternative to using the algorithms we have developed is to solve this problem as a optimization problem. This has been done using Google Optimization Research Tools (aka OR-Tools), which we have implemented in this project. By formulating the optimization problem where the objective function is the minimum weighted vertex cover (WVC), contraints could be added to account for collisions between trajectories and shared donor/target pairs. Since the minimum WVC contains the nodes not in the maximum weighted independent set (WIS), one only needs to solve either one to find the other. It should be noted this does not necessary hold for approximations to the exact solutions, however this is not a concern for our case.

Two methods from OR-Tools have been tested, Mixed Integer Programming (MIP) where the variables are allowed to be arbitrary integers, and an Assignment Problem with contraints. The MIP was initially tested since the exact vertex cover could be set up as such a problem. This worked better than our algorithms, but as the problem is NP-complete, the runtime was higher. Next, since the task of assigning the trajectories to either do or do not, the problem could instead be treated as a Constraint Programming problem, using a Satisfiability method (CP-SAT). This time the variables would either be assigned a 0 or 1 and only one trajectory from each list of collisions could be included in the maximum WIS. This proved to be incredibly effective and solved the exact solution faster than the approximations we had ourself implemented. Seemingly, the runtime is linear with number of trajectories but exponential with number of constraints. Since the number rof contraints is far less than the number of trajectories, this did not make a huge impact on the total runtime

## 6 Testing other packages for implementation
The initial program was built using NetworkX, but other graph tools were considered. The Python module *graph-tool* was quickly shelved as it turned out that it only works for Linux. Next, *igraph* was also tested with the hope that it would decrease the time complexity of the program. Graph building was tested using igraph, and it was twice as fast as using NetworkX. However, igraph is less developed than NetworkX, with less internal functionality. After implementing a few algorithms in igraph and comparing the results to the outcome of equivalent algorithms built with NetworkX, other weaknesses became evident, as the results of algorithms for the exact solution differed, as well as the outcomes of the bipartite matching algorithms. This may reflect flaws in igraph's functions, or be due to different understandings of how a graph should be created before applying the algorithms. Because of this uncertainty and general underdevelopment of igraph, we decided to continue developing the program using NetworkX. To tackle the problematic runtimes of NetworkX, OR-Tools was used with very good results.  

## 7 Testing

### 7.1 Pytest
The automatic tests for our program are run with the pytest framework. The pytest framework is used to write small tests in Python programs. These tests are used through development to showcase potential errors in the program and to show that algorithms and functions run as expected.

Our tests-file includes:
- Tests for simple functions
- Test for algorithms (looking at the algorithms results against results solved by hand)
- Tests for writing and reading to and from json-files
- Tests for timing
- Tests for different command line inputs

All of these tests need to pass for new developments to be merged into the main branch on GitHub.

### 7.2 Solving for algorithms by hand
In the beginning of the project, five small datasets were created, on which all of the upcoming heuristics were to be tested. The first three greedy algorithms and the bipartite matching algorithm were implemented and used on these datasets, and the expected solutions of each of these algorithms were found by hand as a test, and then saved for future comparison. As it turned out, both the greedy algorithm considering value blocked and the one considering number of nodes blocked had been implemented wrongly. The issues were then fixed.

## 8 Results

### 8.1 Value/Performance
The results of a selection of the algorithms described above are shown in the Excel-document in the Appendix section. The algorithms are run on datasets with different combinations of number of donors, targets and trajectories. As one can read in the document, the weight transformation algorithm clearly gives the best results, and that the reversed greedy utilizing weight transformation also is significantly better than the original greedy algorithm. It is also interesting to see that the bipartite matching algorithms actually perform worse than the greedy algorithm.

### 8.2 Runtime
The figure below shows the runtime of the algorithms. Here it is clear that the greedy algorithm and it's variants (such as the weight transformation) are not efficient and demand relatively much time. On the oter hand, the bipartite matching algorithm works fast in comparison to the others. However, this might be because the bipartite matching used is imported from NetworkX, and is therefore expected to be faster than the greedy algorithms we have implemented on our own.
![runtime_analysis](https://user-images.githubusercontent.com/86296731/127311069-48c84c07-6764-4541-b9b4-67663cab91ec.png)

### 8.3 OR-Tools
The best results were however not achieved by using the algorithms and graph theory, but by treating the problem as a Constraint Programming problem, using a Satisfiability method (CP-SAT). This was extremely efficient and solved the exact solution faster than the approximations we had implemented ourself. In other words, Constraint Programming generated the best results in regards to both value and runtime. Given that OR-Tools gives the exact solution, solutions are feasible for datasets with realistic sizes, meaning up to 200 000 trajectories, although high numbers of collisions slow down the runtime. 

![Picture1](https://user-images.githubusercontent.com/43993453/128465362-6e88c60e-e2ea-4ef2-a037-78229d25cc01.png)

## 9 Further ideas
In the end of the internship we had some ideas we unfortunately did not have time to implement.

Firstly, it would have been interesting to make a heuristic which cluster trajectories in space. Here a suggestion is that every donor and target could contain information about where it is located in space by, for instance, x-, y- and z-coordinates. Then the computer would know where each trajectory starts (donor coordinates) and ends (target coordinates). If the donor and target associated with one trajectory is far away from the donor and target associated with another trajectory, the idea is that those two trajectories are unlikely to collide in space. In other words, if one implements a clustering algorithm which clusters trajectories which are relatively close to each other in space, then our algorithms can be run on each of these clusters. This would be under the assumption that trajectories in different clusters don't collide with each other, since they would have different donors and targets, and the donors and targets are far away from each other, so that collisions in space also are highly unlikely. Since all of the clusters now can be considered as independent sets of trajectories, the algorithms could be run on each of these smaller sets, resulting in less computations and lower runtimes.      

Secondly, it would be smart to try out another package from python than networkx, as it seems to take much time. We tried to use igraph instead of networkx, and this needed only minor changes in the code for networkx. The result was lower runtimes. However, the results when using igraph was actually worse than the ones using networkx, and despite the improvements in runtime, we therefore decided not to go on with igraph. However, the improvements show that it could be interesting to try out other graphical tools than networkx.

Thirdly, if one considers to implement the heuristic which cluster trajectories in space, it would be interesting to create a more realistic dataset which actually place trajectories different places in space. This could advantageously be done by assigning a x-, y- and z-coordinate to every donor and target, so that the program knows where in space all trajectories begin and end.

Since no such heuristic has been made yet, no such realistic dataset has been made. However, we have made a function create_realistic_data() which makes a realistic dataset in another way than the one already described. Let it be clear that this function is not very generic and would typically only work well for some inputs of number of donors, number of targets, number of trajectories and collision rate. However, the function is good for it's purpose, since it's purpose is to creta a realistic dataset, and this could be done.

The function creates a dataset of the correct format for the trajectory picking problem. The dataset consists of donors, targets, and trajectories, and a list of collisions. It is in multiple ways attempted to be more realistic than the dataset created from create_data():
- The respective donors and targets of a trajectory is to be imgained as fairly close to each other. This is because the selectable trajectories in a realistic dataset typically will be short, since their cost depends on their length. In this function, the position of donors and targets are implied by their ids, which is sort of imagined to represent it's location. As an example, if there are 50 targets and 50 donors, then trajectories of donor id equal to 15 will typically go to a target id close to 15. However, if there are 50 targets and 200 trajectories, then a trajectory of donor id 15 will typically go to a target id close to 15*(200/50)
- In addition to the location of trajectories, the collisions between them are - in this function - attempted to be generated more realistically. Here it is a five percent chance of each trajectory to collide with another trajectories, a 0.05*0.05 chance to collide with two trajectories etc. Since realistic datasets are of a maximum number of trajectories of approximately 200 000, the chance for any of these trajectories to collide with four or more trajectories is such low that this function is to neglect this.
- If a trajectory collides with other trajectories, then a list of trajectories close to it is made by appending trajectories with donors and targets close to the relevant trajectory. Then the trajectory it collides with is randomly picked from this list, and the collision is appended to a list of collisions. 
 
In other words, the dataset is realistic in the way that it consists of trajectories imagined to be close to each other in space. This is done through donor and targets IDs. In addition, there can only be collisions between trajectories close to each other in space, i.e. their donors and targets are fairly close to each other.

To be able to optimize on an arbitrary key, adding another input argument such as -optimize_on "value"|"risk"


## 10 Conclusion

In sum, the results of the AIM trajectory picking problem has shown that using the greedy algorithm for picking wellbore trajectories that optimize a value is good, but that better alternatives exist. However, better algorithms, such as weight transformation, suffer from poor runtimes. To overcome this problem, the problem was formulated as an linear optimization problem and applied to OR-Tools. This resulted in both high values and incredible runtimes, compared to the algorithms that were implemented using NetworkX and the graph-formulation of the problem. 

## 11 Appendix
- [results_random_datasets.xlsx](https://github.com/equinor/aim-trajectory-picking/files/6938354/results_random_datasets.xlsx)

