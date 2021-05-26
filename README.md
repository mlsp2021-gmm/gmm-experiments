This repository contains a set of experiments for the Gamma Mixture node on Forney-style Factor Graphs.

## Dependencies

Before you can run the experiments you need to have Julia 1.6.x installed on your machine. We use the [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) package to structure our experiments, such that they can be reproduced easily. [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) has to be installed in your global Julia environment before running the experiments. You can add [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) by calling

```
(@v1.6) pkg> add DrWatson
```

or 

```
julia -e 'import Pkg; Pkg.add("DrWatson")
```

We use [`git-lfs`](https://git-lfs.github.com) to commit binaries (e.g. plots and images) in the repository. While it's not required it's highly recommended to have it installed on your machine.

## Instantiating

For reproducability we have fixed the versions of all required packages in this project. To instantiate the project you may use the following command in a terminal:

```
julia --project -e 'import Pkg; Pkg.instantiate()'
```

This command will install all required packages and will prepare the project environment.

## Experiments

All experiments are located in the `experiments` folder. To run an individual experiment you may use the following command in a terminal:

```
julia experiments/<experiment_name>.jl
```

It is also possible to run experiments from any IDE ([Visual Studio Code](https://code.visualstudio.com), [Juno](https://junolab.org), etc...) or within `experiments` folder directly:

```
cd experiments
julia <experiment_name>.jl
```



It is not necessary to activate a project environment before running experiments since `DrWatson.jl` will do this automatically.

## Cached results

Some experiments may take a lot of time to complete. Optionally you can download the `dump.zip` archive from the GitHub releases section, which contains precomputed JLD2 files for the synthetic experiments. By default the experiments pipeline searches for cached results in the `dump` folder and doesn't recompute them if the corresponding cache exists. It is possible to reload the precomputed results and to analyse them in REPL or Visual Studio Code without running all experiments from scratch.

To force the experiments pipeline to recompute results you may either remove the corresponding cached results from the `dump` folder or modify experiments to use the `force = true` flag in the `produce_or_load` method:

```
result, _ = produce_or_load(..., force = true) do params
    run_experiment(params)
end
```

## Project structure

- `data` - datasets of real-world data used for experiments
- `dump` - (optional), cached results of the experiments in JLD2 files
- `experiments` - code/scripts for experiments
- `results` - plots for each individual result, both for the real-world and synthetic datasets
- `src` - reused code, project module, model definitions and utilities
