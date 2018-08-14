# Computing

Nabu has several computing modes. If you train or test a model, you will use
one of these modes to do the computation. You can choose the computation mode
by poining to a configuration file in config/computing in your scripts.
Some of these computing modes are distributed computing modes. For distributed
computing modes you can use multiple devices in parallel. For these modes there
are 2 kinds of jobs:
1) A parameter server job simply holds the parameters of your model and shares
them wit the other devices. Multiple parameter servers are possible, in this
case the parameters will be devide among the servers. A parameter will normally
run on CPU.
2) A worker job reads parameters from the parameter server, does the actual
computation and submits gradients back to the parameter server to update the
parameters. Multiple workers can work in parallel. A worker job normally runs
on a GPU.

## Standard

The standard computing modes do not use a distributed comuting system (Like
HTCondor or Kubernetes) but rely on the user to determine on what machines they
want to run the scripts.

### Non Distributed

The non distributed mode will run on a single device. This is
the simplest mode of computation. you can choose this mode py pointing to
config/computing/standart/non_distributed.cfg. The script will run on the device
its called from.

### Single Machine

The Single machine mode will run on multiple devices within a single machine.
you can choose this mode py pointing to
config/computing/standart/single_machine.cfg. In the configuration you can
choose the amount of parameter servers and workers. It is advised that the
amount of workers does not exceed the number of GPUs in the machine.
The script will run on the device its called from.

### Multi Machine

The Multi machine mode will run on multiple devices that do not need to be in
the same machine. To run this mode you should first create a cluster file.
In the cluster file there should be one line per job. This line contains
the type of job (ps or worker), the adress of the machine, a network port for
communication and a GPU this job should run on (may be empty). An example
cluster file for 2 parameter servers and 3 workers is:

```
ps,ps1.example.com,1024,
ps,ps2.example.com,1024,
worker,worker1.example.com,1024,0
worker,worker1.example.com,1025,1
worker,worker2.example.com,1024,0
```

You can the use this mode by pointing to
config/computing/standart/multi_machine.cfg. In this config you should point to
the cluster file you created.

## Condor

The Condor compute modes use [HTCondor](https://research.cs.wisc.edu/htcondor/)
to select the machines to run the scripts on instead of relying on the user to do
this. The same computing modes are possible with Condor as he standar compute
modes. The configurations can be found in config/computing/condor. Before you
can start using condor you should create the create_environment.sh script in the
nabu/computing/condor/ directory. This file should create the necesarry
environment variables (like PYTHONPATH) and then execute its arguments. An
example script:

```
#!/bin/sh

#create the necesary environment variables
export PYTHONPATH=/path/to/python/libs

#execute arguments
$@
```
