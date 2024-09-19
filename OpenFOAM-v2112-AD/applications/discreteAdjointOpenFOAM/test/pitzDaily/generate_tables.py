#! /usr/bin/env python3

from tabulate import tabulate
from sys import argv

try:
    if len(argv) < 2:
        raise(Exception("No filename given"))

    timings = {}
    with open(argv[1]) as f:
        for l in f:
            t = l.strip().split()
            # print(t)
            timings[t[0]] = [float(t[1]), 1.0]
            #timings.append( (t[0], float( t[1]), 1.0 ) )

    # passive primal to activated primal
    timings["simpleFoam.linux64GccDPInt64OptA1SDCO_FOAM"][1] = timings["simpleFoam.linux64GccDPInt64OptA1SDCO_FOAM"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["simpleFoam.linux64GccDPInt64OptA1SCODI"][1] = timings["simpleFoam.linux64GccDPInt64OptA1SCODI"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["simpleFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][1] = timings["simpleFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]


    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]

    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"][0]

    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI_parallel"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI_parallel"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM_parallel"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM_parallel"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM_parallel"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM_parallel"][0]
    timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV_parallel"][1] = timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV_parallel"][0] / timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM_parallel"][0]

    #for k,v in timings.items():
    #    print(k,v)

    t = []
    for k,v in timings.items():
        t.append([k, v[0], v[1]])

    table1 = []
    table1 += [["Passive Primal"] + timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"]]
    table1 += [["Activated Primal dco_foam"] + timings["simpleFoam.linux64GccDPInt64OptA1SDCO_FOAM"]]
    table1 += [["Activated Primal dco_cpp_dev"] + timings["simpleFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"]]
    table1 += [["Activated Primal codi"] + timings["simpleFoam.linux64GccDPInt64OptA1SCODI"]]

    table2 = []
    table2 += [["Passive Primal"] + timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM"]]
    table2 += [["Checkpointed solver dco_foam"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM"]]
    table2 += [["Checkpointed solver dco_cpp_dev"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV"]]
    table2 += [["Checkpointed solver codi"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI"]]

    table3 = []
    table3 += [["Passive Parallel Primal"] + timings["simpleFoam.linux64GccDPInt64OptPassiveDCO_FOAM_parallel"]]
    table3 += [["Checkpointed solver dco_foam"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_FOAM_parallel"]]
    table3 += [["Checkpointed solver dco_cpp_dev"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SDCO_CPP_DEV_parallel"]]
    table3 += [["Checkpointed solver codi"] + timings["adjointSimpleCheckpointingLambdaFoam.linux64GccDPInt64OptA1SCODI_parallel"]]

    colwidth=45
    print(tabulate(table1,headers=["Activated Primal vs. Passive Primal".ljust(colwidth),"time","factor"],tablefmt="fancy_grid",floatfmt=".2f"))
    print(tabulate(table2,headers=["Checkpointed Adjoint vs. Passive Primal".ljust(colwidth),"time","factor"],tablefmt="fancy_grid",floatfmt=".2f"))
    print(tabulate(table3,headers=["Parallel Checkp. Adjoint vs. Passive Primal".ljust(colwidth),"time","factor"],tablefmt="fancy_grid",floatfmt=".2f"))
except Exception as e:
    print(e)