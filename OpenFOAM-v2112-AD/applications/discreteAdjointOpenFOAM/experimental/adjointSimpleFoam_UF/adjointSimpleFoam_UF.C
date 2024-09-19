/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    simpleFoam

Group
    grpIncompressibleSolvers

Description
    Steady-state solver for incompressible flows with turbulence modelling.

    \heading Solver details
    The solver uses the SIMPLE algorithm to solve the continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

\*---------------------------------------------------------------------------*/

#include "profiling.H"

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

#include "costFunctionLibrary.H"

//#include "dco.hpp"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// make sure correct environment is sourced
// static_assert ( std::is_same<scalar, ADmode::type>::value , "AD mode must be a1s");

int main(int argc, char *argv[])
{
    #include "postProcess.H"

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFvOptions.H"
    #include "initContinuityErrs.H"

    #include "adjointSettings.H"

    turbulence->validate();

    label tapeSizeMB = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<label>("tapeSizeMB",4096);
    Info << "Creating Tape, size: " << tapeSizeMB << endl;
    AD::createGlobalTape(tapeSizeMB/Pstream::nProcs());

    label nCells = U.size();
    label nFaces = phi.size();

    std::vector<double> Jacobian(2*nCells*nFaces, 0.0);

    AD::registerInputVariable(Uf.begin(), Uf.end());
    phi += Uf;

    // --- Pressure-velocity SIMPLE corrector
    {
        #include "UEqn.H"
        #include "pEqn.H"
    }

    laminarTransport.correct();

    // turbulence correct
    turbulence->correct();
    AD::switchTapeToPassive();

    for(int i=0; i<nCells; i++){
        for(int k=0; k<2; k++){
            Info << "Interpret " << i << " / " << nCells << endl;
            AD::derivative(U[i][k]) = 1.0;
            AD::interpretTape();
            for(int j=0; j<nFaces; j++){
                Jacobian[(2*i+k)*nFaces+j] = AD::derivative(Uf[j]);
            }
            AD::zeroAdjointVector();
        }
    }
    std::ofstream ofs("Jac");
    for(int i=0; i<2*nCells; i++){
        for(int j=0; j<nFaces; j++){
            ofs << Jacobian[i*nFaces+j] << " ";
        }  ofs << "\n";
    }
    ofs.close();

    AD::resetTape();

    Foam::profiling::print(Info);
    ADmode::tape_t::remove(ADmode::global_tape);
    Info<< "End\n" << endl;
    return 0;
}


// ************************************************************************* //
