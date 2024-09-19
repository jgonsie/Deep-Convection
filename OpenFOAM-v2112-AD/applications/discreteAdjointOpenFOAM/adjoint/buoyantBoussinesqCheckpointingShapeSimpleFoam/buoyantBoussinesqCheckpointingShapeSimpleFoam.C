/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
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
    buoyantBoussinesqSimpleFoam

Group
    grpHeatTransferSolvers

Description
    Steady-state solver for buoyant, turbulent flow of incompressible fluids.

    Uses the Boussinesq approximation:
    \f[
        rho_{k} = 1 - beta(T - T_{ref})
    \f]

    where:
        \f$ rho_{k} \f$ = the effective (driving) density
        beta = thermal expansion coefficient [1/K]
        T = temperature [K]
        \f$ T_{ref} \f$ = reference temperature [K]

    Valid when:
    \f[
        \frac{beta(T - T_{ref})}{rho_{ref}} << 1
    \f]

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "radiationModel.H"
#include "fvOptions.H"
#include "simpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "costFunctionLibrary.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// make sure correct environment is sourced
static_assert ( std::is_same<scalar, ADmode::type>::value , "AD mode must be a1s");

#include "CheckController.H"
#include "CheckInterface.H"
#include "CheckDict.H"
#include "CheckController.H"

#include <functional>

template<class RunStep, class CalcCost, class WriteSens>
class SimpleLambdaCheckpointer : public CheckController{
private:
    const RunStep& runStepFunc;
    const CalcCost& calcCostFunc;
    const WriteSens& writeSensFunc;
public:
    SimpleLambdaCheckpointer
    (
        Foam::Time& runTime,
        const RunStep& runStepFunc,
        const CalcCost& calcCostFunc,
        const WriteSens& writeSensFunc
    )
    :
        CheckController(runTime),
        runStepFunc(runStepFunc),
        calcCostFunc(calcCostFunc),
        writeSensFunc(writeSensFunc)
    {
    }

    bool runStep(){
        return runStepFunc();
    }

    scalar calcCost(){
        return calcCostFunc();
    }

    void postInterpret(){writeSensFunc();}
    void write(bool firstRun){writeSensFunc();}
    void start(){}

    auto& checkDB(){
        return checkInterface().checkDatabase();
    }
};

void initMesh(Foam::fvMesh& mesh){
    mesh.Sf();
    mesh.magSf();
    mesh.C();
    mesh.Cf();
    mesh.V();
    mesh.deltaCoeffs();
    mesh.nonOrthDeltaCoeffs();
    mesh.nonOrthCorrectionVectors();
}

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Steady-state solver for buoyant, turbulent flow"
        " of incompressible fluids."
    );

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    
    AD::createGlobalTape(32000);
    // register mesh points
    #include "createMesh.H"
    
    
    #include "createControl.H"
    #include "createFields.H"
    #include "initContinuityErrs.H"

    turbulence->validate();
    
    initMesh(mesh);
    AD::switchTapeToPassive();

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    auto runStep = [&]() -> bool{
        // advance runTime and return if at endTime or converged
        if(!simple.loop())
        {
            return true;
        }
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "TEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
        turbulence->correct();

        runTime.write();

        runTime.printExecutionTime(Info);

        return false;
    };

    auto calcCost = [&]() -> scalar {
        scalar J=0;
        return J;
    };

    auto writeSens = [&]() {

    };

    SimpleLambdaCheckpointer checkpointer(runTime,runStep,calcCost,writeSens);
    checkpointer.run();

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
