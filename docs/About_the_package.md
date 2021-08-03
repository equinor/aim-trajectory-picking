# Equinor Summer Intern AIM Project 2021

## 1 Introduction
This report includes discoveries, reflections and results for the AIM trajectory picking project, carried out by five summer interns. The goal of the project was to explore and discover algorithms that coordinate wellbore trajectories and pick paths that optimize different variables, given certain constraints.
Knowing from the start that the greedy algorithm performs well on the given problem despite its limitations, it became the baseline algorithm which the aim was to beat. 

## 2 Setting up the problem
To operationalize the trajectory picking problem, trajectories were considered nodes in a graph. Given this formulation, ideas and solutions from graph theory could be utilized to solve the problem. Each trajectory has a donor, a target and a non-negative value, in addition to a list of other trajectories it collides with. Given that no trajectories should share the same donor, nor target, and not collide with other trajectories, the task was to pick trajectories that maximize the sum of trajectory values. 

## 3 Algorithms
During the working prosess we developed multiple modifications of the greedy algorithm and new algorithms following other heuristics.

### 3.1 Greedy
Many different variants of the greedy algorithm has been implemented. If some of these were to be an improvement of the original greedy algorithm, then this would be an easy change of proven improvements.

#### 3.1.1 The Original Greedy
First the greedy algorithm was implemented. Here the Python package "networkx" was used to create a graph where each node represents a trajectory. The edges between these nodes imply that the current trajectories are mutually exclusive, i.e. the trajectories either have the same donor, same target, or collide with each other. Then the greedy algorithm was implemented to pick the node of highets value, remove the neighbors of this node which now cannot be picked, and then choose the node of highest value of the remaining nodes etc. This is done until there is no nodes left to pick, and the total value is the sum of the values of all nodes picked.

#### 3.1.2 Greedy Considering Value Blocked
One of the main weaknesses of the greedy algorithm described above is that it only concerns about picking nodes of high value. However, to maximize the total value of trajectories picked, we do not only want nodes of high value, but also many nodes. For the greedy algorithm to take both of these two factors into account, the greedy algorithm considering value blocked was implemented. This algorithm is very similar to the greedy above; the only difference is that instead of picking the node of highest value at all times, this algorithm instead picks the node which fraction, equal to value divided by the sum of value blocked of neighbors, is highest. 

#### 3.1.3 Greedy Considering Number of Nodes Blocked
This algorithm is motivated by the same factors in which the greedy considering value blocked was motivated. Instead of using the sum of value blocked in the denominator, the denominator in this algorithm equals the number of nodes blocked, plus 1. The extra "1" is added to avoid a possible zero in the denominator. 

#### 3.1.4 Reversed Greedy
All of the three greedy variants explained above are strategies with the intention of picking "good" nodes from a set of nodes. The reversed greedy, however, focus on eliminating "bad" nodes blocking many other nodes. This algorithm uses the greedy algorithm to eliminate these nodes. When the "worst" nodes thereby are removed, there are three ways in which the rest of the problem can be solved. The first and second methods are by using the original greedy or the greedy considering value blocked. The third way is to solve for the remaining collisions by using the greedy algorithm, and then solve for donors and targets by bipartite matching.

#### 3.1.5 Greedy Focusing on Lonely Targets
The idea behind this algorithm is that when we hit many donors the probability of getting a high total value is increased. Even though all the trajectories are not chosen based on value. This algorithm firstly focuses on choosing all the targets only hit by one trajectory and then use the greedy. This way we chose all the targets that remove the least amount of other trajectories before starting the regular greedy algorithm.

### 3.2 Random Choice
This algortihm randomly chooses it's nodes in every step of the greedy algorithm, not considering their value nor blockages. It was implemeted to get an intuition of how good the greedy algorithm picking nodes of high value actually is.

### 3.3 Bipartite Matching
The bipartite matching algorithm uses bipartite matching to pick trajectories. In this case a graph where each node represents a trajectory is still used, but with edges only for collisions; they do not take collision in donors or targets into account. Here the greedy algorithm is used to pick a set of nodes which will only contain non-colliding trajectories. Now that the collisions are taken care of, the challenge of donors and targets is solved by making a bipartite graph where the two independent sets of nodes are respectively donors and targets, and the edges between them illustrates the trajectories. Therefore the edges are weighted by value. A built-in function in networkx is then used to perform the bipartite matching on this set of nodes, and the resulting list of nodes are the ones picked by this strategy.

### 3.4 Bipartite Matching, version 2
This bipartite matching algorithm is very similar to the one described above. The only difference is that, instead of solving for collisions by using the greedy algorithm, it uses the reversed greedy algorithm

### 3.5 Reversed Bipartite Matching
This algorithm is similar to the Bipartite Matching algorithm, but works in the opposite direction. This function first solves for donors and targets by using bipartite matching, and afterwards solves for collisions. If any of the trajectories chosen from the bipartite matching collides, then the ones of lowest value are removed. Then, the algorithm looks for the trajectories of highest value which can replace the removed trajectories.

## 4 Alternatives (OR-Tools)
An alternative to the algorithms we have developed is to solve this problem by integer-linear-programming. This could be done by usng OR-Tools which we have implemented in this project. This worked better than our algorithms, but on the other hand, the runtime was higher. However, for small problems with sufficiently few trajectories, we recommend ILP to be used.

## 5 Exact solutions

### 5.1 Maximum (Weighted) Independent Set
The problem of picking the best trajectories can be considered a maximum weighted independent set problem, and is a so called strictly NP-hard problem. Since there are no solution to such a problem in polynomial time, it would be unrealistic to use this algorithm for a full sized dataset. Instead, testing has been done to determine how large of a dataset one could reasonably use such an algorithm on. \todo{må oppdatere her underveis som vi tester denne algoritmen} Other approaches could be to solve the minimum weighted vertex cover and take the trajectories not in the vertex cover or the maximum weighted clique problem on the complimentary graph. These are also so called NP-hard problems, so the runtime may not see much improvement when solved exact. The maximum clique algorithm from Networkx

### 5.2 Approximations to the MIS algorithm
Approximations of the afore mentioned problems may result in exact or near-exact results at a fraction of the runtime that the exact methon would use. 

## 6 Implementation
The program was built using NetworkX, but other graph tools were considered. The Python module \textit{graph-tool} was quickly shelved as it turned out that it only works for Linux. Next, \textit{igraph} was also tested with the hope that it would decrease the time complexity of the program. Graph building was tested using igraph, and it was twice as fast as using NetworkX. However, igraph is less developed than NetworkX, with less internal functionality. After implementing a few algorithms in igraph and comparing the results to the outcome of equivalent algorithms built with NetworkX, other weaknesses became evident, as the results of algorithms for the exact solution differed, as well as the outcomes of the bipartite matching algorithms. This may reflect flaws in igraph's functions, or be due to different understandings of how a graph should be created before applying the algorithms. Because of this uncertainty and general underdevelopment of igraph, we decided to continue developing the program using NetworkX.

## 7 Testing

### 7.1 Pytest
The automatic tests for our program are run with the pytest framework. The pytest framework is used to write small tests in python programs. These tests are used through development to showcase potential errors in the program and to show that algorithms and functions run as expected.

Our tests-file includes:
- Tests for simple functions

- Test for algorithms (looking at the algorithms result against results solved by hand)

- Tests for writing and reading to and from json-files

All of these tests need to pass for new developments to be merged into the main branch on GitHub.

### 7.2 Solving for algorithms by hand
In the beginning of the project, five small datasets were made, on which all of the upcoming heuristics were to be tested. The first three greedy algorithms and the bipartite matching were implemented and used on these datasets, and the expected solutions of each of these algorithms were found by hand as a test, and then aved for future comparison. As it turned out, both the greedy algorithm considering value blocked and the one considering number of nodes blocked had been implemented wrongly. The issues were then fixed.

## 8 Results

### 8.1 Value/Performance
The results of a selection of the algorithms described above are shown in the excel-document in the Appendix section. The algorithms are run on datasets with different combinations of number of donors, targets and trajectories. Here one can see that the weight transformation algorithm clearly gives the best results, and that the reversed greedy utilizing weight transformation also is significantly better than the original greedy algorithm. It is also interesting to see that the bipartite matching algorithms actually perform worse than the greedy algorithm. The results are discussed in more detail in the discussion section.

### 8.2 Runtime
The figure below shows the runtime of the algorithms. Here it is clear that the greedy algorithm and it's variants (such as the weight transformation) are not efficient and demand relatively much time. On the oter hand, the bipartite matching algorithm works fast in comparison to the others. However, this might be because the bipartite matching used is imported from networkx, and is therefore expected to be faster than the greedy algorithms we have implemented on our own.
![runtime_analysis](https://user-images.githubusercontent.com/86296731/127311069-48c84c07-6764-4541-b9b4-67663cab91ec.png)

## 9 Further ideas
In the end of the internship we had some ideas we unfortunately did not have time to implement.

Firstly, it would have been interesting to make a heuristic which cluster trajectories in space. Here a suggestion is that every donor and target could contain information about where it is located in space by, for instance, x-, y- and z-coordinates. Then the computer would know where each trajectory starts (donor coordinates) and ends (target coordinates). If the donor and target associated with one trajectory is far away from the donor and target associated with another trajectory, the idea is that those two trajectories are not likely to collide in space. In other words, if one implements a clustering algorithm which clusters trajectories which are relatively close to each other in space, then our algorithms can be run on each of these clusters. This would be under the assumption that trajectories in different clusters don't collide with each other, since they would have different donors and targets, adn the donors and targets are far away from each other, so that collisions in space also are highly unlikely. Since all of the clusters now can be considered as independent sets of trajectories, the algorithms could be run on each of these smaller sets, resulting in less computations and lower runtimes.      

Secondly, it would be smart to try out another package from python than networkx, as it seems to take much time. We tried to use igraph instead of networkx, and this needed only minor changes in the code for networkx. The result was lower runtimes. However, the results when using igraphs was actually worse than the ones using networkx, and despite the improvements in runtime, we therefore decided not to go on with igraph. However, the improvements show that it could be interesting to try out outher graphical tools than networkx.

Thirdly, if one considers to implement the heuristic which cluster trajectories in space, it would be interesting to create a more realistic dataset which actually place trajectories different places in space. This could advantageously be done by assigning a x-, y- and z-coordinate to every donor and target, so that the program knows where in space all trajectories begin and end.

Since no such heuristic has been made yet, no such realistic dataset has been made neither. However, we have made a function create_realistic_data() which makes a realistic dataset in another way than the one already described. Let it be clear that this function is not very generic and would typically only work well for some inputs of numbers of donors, number of targets, number of trajectories and collision rate. However, the function is good for it's purpose, since it's purpose is to creta a realistic dataset, and this could be doneCreates a dataset of the correct format for the trajectory picking problem.

The dataset consists of donors, targets, and trajectories, and a list of collisions. It is in multiple ways attempted to be more realistic than the dataset created from create_data():

- The respective donors and targets of a trajectory is to be imgained as fairly close to each other. This is because the selectable trajectories in a realistic dataset typically will be short, since the cost of them depends on their length. I this function, the position of donors and targets are implied by their ids, which is sort of imagined to represent it's location. As an example, if there are 50 targets and 50 donors, then trajectories of donor id qual to 15 will typically go to a target id close to 15. However, if there are 50 targets and 200 trajectories, then a trajectory of donor id 15 will typically go to a target id close to 15*(200/50)

- In addition to the locaion of trajectories, the collisions between them are - in this function - attempted to be generated more realistically. Here it is a five percent chance of each trajectory to collide with another trajectories, a 0.05*0.05 chance to collide with two trajectories etc. Since realistic datasets are of a maximum number of trajectories of approximately 200 000, the chance for any of these trajectories to collide with four or more trajectories is such low that this function decides neglect this.

- If a trajectory collides with other trajectories, then a list of trajectories close to it is made by appending trajectories with donors and targets close to the relevant trajectory. Then the trajectory it collides with is randomly picked from this list, and the collision is appended to a list of collisions. In other words, the dataset is realistic in the way that it consists of trajectories imagined to be close to each other in space. This is done through donor and targets ids. In addition, there can only be collisions between trajectories close to each other in space, i.e. their donors and targets are fairly close to each other.

## 10 Discussion

## 11 Conclusion

## 12 Appendix
[resultater_for_mange_random_sett.xlsx](https://github.com/equinor/aim-trajectory-picking/files/6892701/resultater_for_mange_random_sett.xlsx)
