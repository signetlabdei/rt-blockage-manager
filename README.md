# RT-blockage Manager
A Python open-source framework that interfaces with the ray tracer [Q-D Realization](https://github.com/signetlabdei/qd-realization/) to include the effect of multiple dynamic obstacles in the simulations. The RT-Blockage Manager offers a simple yet powerful interface to set up simulations with a minimal amount of code, processing the RT output traces by introducing the effect of the blockers on each ray.
A brief description and the usage of this software can be found in the following paper:

- [P. Testolina, M. Lecci, A. Traspadini, M. Zorzi, "An Open Framework to Model Diffraction by Dynamic Blockers in Millimiter Wave Simulations," in IEEE 20th Mediterranean Communication and Computer Networking Conference (MedComNet), Paphos, Cyprus, Jun. 2022.](https://ieeexplore.ieee.org/document/9810361)

Different diffraction models have already been implemented to precisely model the effect of blockers to the millimiter wave (mmWave) signal.


## Table of Contents
* [Installation](#installation)
* [How to Run](#how-to-run)
* [Software Modules](#software-modules)
* [CircleCI](#circleci)
* [Authors](#authors)

## Installation
The repository can be cloned or downloaded to your local directory:

```bash
git clone https://github.com/signetlabdei/rt-blockage-manager
```

Then, you need to install the required Python libraries:

```bash
pip install -r rt-blockage-manager/requirements.txt
```

## How to Run

The first step is to run the RT to obtain the traces for the static scenario, more details about this software can be found in its [repository](https://github.com/signetlabdei/qd-realization/).
The RT-Blockage Manager uses the static output as the baseline in which the effect of multiple dynamic blockers is introduced, this allows running the highly-computational demanding traces only once.
The RT traces can be imported in the RT-Blockage manager using the general `Scenario` interface, already extended into `QdRealizationScenario` to handle the traces obtained from the [Q-D Realization](https://github.com/signetlabdei/qd-realization/), which is specifically able to handle multiple users and timesteps.
Then, blockers can be introduced in the scenario with the `Obstacle` interface, in which some general classes are implemented:

* `SphereObstacle` to represent a simple obstacle with a fixed transmission loss affecting the path gain.

* `ScreenObstacle` to represent a general thin screen obstacle with an arbitrary 3D orientation. 

* `OrthoScreenObstacle` to represent a thin screen obstacle with an orientation always orthogonal to the direction of propagation, valid for a larger variety of diffraction models.

The movement of an obstacle during the simulation is described by a `MobilityModel`. At each time step, the position of each obstacle is updated based on such model, thereby providing accurate and temporally-correlated mobility and making the channel temporally consistent.
The whole simulation can be run with just a few lines of code through the class `Environment`, which receives as parameters the `Scenario` and the `Obstacle`, to process the interaction of the rays with the blockers. 

The script `example environment_example.py` is provided to better understand how to run this software.

## Software Modules

In this section, we briefly present the main modules of the software, designed according to the object-oriented paradigm and briefly described in the following paragraphs.

* `Geometry`
This module contains the geometry notions essential to handle the ray geometry, to define consistent mobility patterns of the obstacle, and to describe the geometry of the obstacles themselves (e.g., *Point*, *Vector*, *Line*, *Segment*, *Plane*, *Rectangle*, *TransfMatrix* (Homogenous Transformation Matrix)). All the basic geometrical operations have been implemented (e.g., projections, dot and cross products, etc.). The classes have been implemented in a user-friendly manner to minimize code writing and maximize clarity. Thanks to the modularity of the software, it is possible to optimize the geometrical operations independently of the other modules.

* `Ray`
This module organizes the information provided by a ray-tracer, such as delay, path gain, phase, and the actual path (i.e., the vertices of the ray-traced path). It also offers a simple interface to consistently compute Angles of Departure (AoDs) and Angles of Arrival (AoAs). Notice that, for the correct import and usage of the Blockage Manager software, the qd-realization software should also export visualization files, namely, *MpcCoordinates*.

* `Scenario`
A *Scenario* interface is defined, which subclasses have to implement. This was done in an attempt to generalize the definition of a scenario, in principle allowing for the possibility to support different ray-tracing formats in the future. The interface defines common methods to import/export traces in the target format, as well as to access and update sets of rays between nodes. The *Scenario* interface was extended into *QdRealization Scenario*, which is specifically able to handle channel traces for multiple users and timesteps.
Notice that *QdRealizationScenario* supports the version of [Q-D Realization](https://github.com/signetlabdei/qd-realization/) used in [Lecci21](https://ieeexplore.ieee.org/document/9459462), not the current master branch. Specifically, JSON outputs are currently not supported, but should be easily included in the future.

* `Obstacle`
A common *Obstacle* interface is defined to handle obstructions, diffraction, and other effects that a generic obstacle may impose over *Rays* from the imported *Scenario*. Currently, a sphere, a rectangular and an orthogonal-rectangular screens are implemented. We defined the orthogonal-rectangular obstacle as an ideal rectangular screen that behaves as if it were orthogonal to any considered ray, when computing the interaction between the two. This artificial obstacle was introduced to meet the hypotheses of several diffraction models, that could thus be included in the software. On the contrary, the rectangular screen can be tilted in both the azimuth and elevation directions, thus providing a more general obstacle mobility at the cost of a limited set of available diffraction models. 

* `Mobility Models`
A user can move an *Obstacle* during a simulation by using *MobilityModels*. Each obstacle will then update its position based on such model, thereby providing an accurate and temporally-correlated mobility and making the channel temporally consistent.
It is important to notice that, at the moment, the Blockage Manager software does not import the CAD scene on which the channel was generated, and mobility models thus ignore fixed obstacles. This means that the obstacle trajectory is generated with no knowledge of the CAD topology: it is a responsibility of the user to make sure it is consistent with the CAD data, if required by the simulation.

* `Diffraction Models`
The *Diffraction Models* module implements different diffraction models as static methods. More details on the implemented diffraction models can be found in [Testolina22](https://ieeexplore.ieee.org/document/9810361).

* `Environment`
It represents the core of the Blockage Manager software, where *Obstacle*s interact with pre-computed *Ray*s. Different kinds of interactions of the rays with the obstacles can be selected: *ObstacleInteraction.OBSTRUCTION*, *ObstacleInteraction.DIFFRACTION* and *ObstacleInteraction.REFLECTION*. The user can thus easily select how the rays behave when interacting with the obstacles. Currently, only diffraction and obstruction are supported.


## CircleCI

we set up a Continuous Integration and Developement (CI/CD) pipeline based on CircleCI 5 , which allows us to fully test our code base every time we push our updated GitHub repository, available [![here](https://circleci.com/gh/signetlabdei/rt-blocakge-manager.svg?style=shield&circle-token=<154c2fd537a19d280b41d416abd54feb3ef6962e>)](https://circleci.com/gh/signetlabdei/rt-blocakge-manager)

## Authors
The RT-Blockage Manager has been developed by the university of Padova, Department of Information Engineering, [SIGNET group](http://signet.dei.unipd.it/).
Besides, it was partially supported by the National Institute of Standards and Technology (NIST).
