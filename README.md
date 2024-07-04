
This is the source code for the program developed for the paper ["Explaining a Staff Rostering Problem using Partial Solutions".](https://doi.org/10.1145/3638530.3654318)

(As of 28/06/2024 I'm working on cleaning up the repository)

The repository consists of
* code from the paper I expanded upon (contained in the directory `FirstPaper`)
	* [Mining Potentially Explanatory Patterns via Partial Solutions](https://arxiv.org/abs/2404.04388)
* code for the new approach (more information below)
* The files to construct the Staff Rostering problem definition in the folder `resources/BT/MartinsInstance`
	* This is the same problem instance used by Martin Fyvie in his many works, eg [this](https://dl.acm.org/doi/10.1007/978-3-031-47994-6_27)
* some files to carry out the statistical testing and plotting in the directory `TestsAndPlots`
* A description of the rota patterns used in the benchmark problems in `resources/BT/benchmark_problems_for_staff_rostering.pdf`


To run this program, you will need the following libraries:
* `numpy`, `pandas` for maths
	* numpy 2.0.0 gives some problems with Pymoo, so you might have to use version 1.26.4
* `numba` to speed up calculations
* `matplotlib`, `plotly` visualisations of plots
* `networkx` visualisations of networks
* `pymoo` Multi-objective algorithms


# General behaviour of the program

The program does the following:
1. given a problem definition, it solves it using SA, GA or uniform sampling
2. While solving the problem, it also generates other data (the reference population) that can be used for explainability
	3. the file is called `pref.npz`
3. The reference population is analysed and Partial Solutions are obtained
	4. they are stored in `pss.npz`
4. The Partial Solutions are analysed and their properties are stored
	5. the properties are in `properties.csv`, and the control PSs are stored in `control.npz`
5. Once all the cached data is obtained (reference population, partial solutions, properties etc..), the explanations can be produced.

You're probably here for the explanations, which you will find by simply running the program and pressing `s` (for solution) and `0` (to select the best one). For more information, type `help`.


I already ran the first 4 steps, so you'll get explanations right away (it is all stored in the folder `ExplanatoryCachedData/StaffRosteringProblemCache`), but if you want to change anything you'll have to re-run them (which is very slow...).

# Other problems
In case you want to play around with other problems, you can see what is available from the folder `BenchmarkProblems`. Once you have decided the problem, you'll have to instantiate it and construct an explainer.

    Explainer.from_folder(problem=problem,  
                          folder=cache_directory,  
                          polarity_threshold=0.10,  
						  verbose=True)

  In case you want to see how the system behaves with your own problem, you'll have to implement it as an instance of BenchmarkProblem, with the implementation of the following methods:
* \_\_init\_\_ (search_space: SearchSpace)
* \_\_repr\_\_() -> str
* fitness_function(fs: FullSolution) -> float
* ps_to_properties(ps: PS) -> dict[str, float]

Optionally, to make the output nicer:
* repr_ps(ps: PS) -> str
	* a pretty printer for Partial Solutions
* repr_descriptor(descr_name, descr_value, polarity, ps) -> str
	* a pretty printer for a "[ps] has [descriptor] = [value], polarity = [polarity]"

Have fun!
