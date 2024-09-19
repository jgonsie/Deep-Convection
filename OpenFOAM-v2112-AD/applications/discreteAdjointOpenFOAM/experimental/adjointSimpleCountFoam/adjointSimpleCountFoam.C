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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// make sure correct environment is sourced
// static_assert ( std::is_same<scalar, ADmode::type>::value , "AD mode must be a1s");

std::vector<std::tuple<std::string,double,double>> nodes;

void print_tape_delta(Time &runTime, std::string varName="dummy"){
    static int i=0;
    static label p = AD::getTapePosition()._progvarcounter();
    static label s = AD::getTapePosition()._stackcounter();
    static scalar t = runTime.elapsedCpuTime();

    int stackSize = AD::getTapePosition()._progvarcounter() - p;
    int tapeSize = AD::getTapePosition()._stackcounter() - s;
    
    /*Info << "tape_" << i << " " 
         << stackSize << " " 
         << tapeSize << " " 
         << runTime.elapsedCpuTime()-t << endl;*/

    nodes.push_back({varName,tapeSize,stackSize});

    i++;
    p = AD::getTapePosition()._progvarcounter();
    s = AD::getTapePosition()._stackcounter();
    t = runTime.elapsedCpuTime();
}

void printNodes(){
    double tapeSize = AD::getTapePosition()._stackcounter();
    std::cout << "m4_divert(-1)" << std::endl;
    for(auto &node : nodes){
        std::cout << "m4_define("
                  << std::get<0>(node) << "," 
                  << std::setw(2) << std::fixed << std::setprecision(3) 
                  << (std::get<1>(node)/tapeSize)*100 << "%)"
                  << std::endl;
    }
    std::cout << "m4_divert(0)" << std::endl;
}

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

    turbulence->validate();

    label tapeSizeMB = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<label>("tapeSizeMB",4096);
    Info << "Creating Tape, size: " << tapeSizeMB << endl;
    AD::createGlobalTape(tapeSizeMB/Pstream::nProcs());

    forAll(U,i){
        AD::registerInputVariable(U[i][0]);
        AD::registerInputVariable(U[i][1]);
        AD::registerInputVariable(U[i][2]);
    }
    AD::registerInputVariable(p.begin(), p.end());
    AD::registerInputVariable(phi.begin(), phi.end());

    print_tape_delta(runTime); // 0

    // don't need the time loop
    //#include "UEqn.H"

    // preparation function f_U
    fvVectorMatrix UEqn
    (
        fvm::div(phi, U)
      + MRF.DDt(U)
      + turbulence->divDevReff(U)
      /*+ fvm::Sp(alpha, U)*/
     ==
        fvOptions(U)
    );

    UEqn.relax();
    fvOptions.constrain(UEqn);

    print_tape_delta(runTime, "m4_UEqn"); // 1

    fvVectorMatrix fullUEqn(UEqn == -fvc::grad(p));

    print_tape_delta(runTime, "m4_fullUEqn"); // 2

    solve(fullUEqn);
    fvOptions.correct(U); // no-op

    print_tape_delta(runTime, "m4_solveU"); // 3

    //#include "pEqn.H"
    volScalarField rAU(1.0/UEqn.A());

    print_tape_delta(runTime, "m4_rAU"); // 4
    
    volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
    
    print_tape_delta(runTime, "m4_HbyA"); // 5

    surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));
    MRF.makeRelative(phiHbyA);
    adjustPhi(phiHbyA, U, p);

    print_tape_delta(runTime, "m4_phiHbyA"); // 6

    // Update the pressure BCs to ensure flux consistency
    constrainPressure(p, U, phiHbyA, rAU, MRF);

    //print_tape_delta(runTime);

    // assume no Non-orthogonal corrections
    fvScalarMatrix pEqn
    (
        fvm::laplacian(rAU, p) == fvc::div(phiHbyA)
    );

    pEqn.setReference(pRefCell, pRefValue);

    print_tape_delta(runTime, "m4_pEqn"); // 7

    // solver call for s_p
    pEqn.solve();
    // Explicitly relax pressure for momentum corrector
    p.relax();

    print_tape_delta(runTime, "m4_solveP"); // 8

    // post processing function f_post
    phi = phiHbyA - pEqn.flux();

    print_tape_delta(runTime, "m4_phi1"); // 9

    // Momentum corrector
    U = HbyA - rAU*fvc::grad(p);
    U.correctBoundaryConditions();
    fvOptions.correct(U);

    print_tape_delta(runTime, "m4_U2"); // 10

    Foam::profiling::print(Info);

    printNodes();

    return 0;
}


// ************************************************************************* //
