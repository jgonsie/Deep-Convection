#! /bin/bash

wmake $FOAM_APP/utilities/mesh/generation/blockMesh
wmake $FOAM_APP/utilities/mesh/manipulation/mirrorMesh
wmake $FOAM_APP/solvers/basic/potentialFoam
wmake $FOAM_APP/discreteAdjointOpenFOAM/mesh/adjointMoveMesh
wmake $FOAM_APP/discreteAdjointOpenFOAM/experimental/piggyShapeSimpleFoam
