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
//static_assert ( std::is_same<scalar, ADmode::type>::value , "AD mode must be a1s");

#include "CheckController.H"
#include "CheckInterface.H"
#include "CheckDict.H"
#include "CheckController.H"


template<class RunStep, class CalcCost, class WriteSens>
class simpleCheckpointer : public CheckController{
private:
    const RunStep& runStepFunc;
    const CalcCost& calcCostFunc;
    const WriteSens& writeSensFunc;
public:
    simpleCheckpointer
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

/*void init_mesh(Foam::fvMesh& mesh){
    mesh.Sf();
    mesh.magSf();
    mesh.C();
    mesh.Cf();
    mesh.V();
    mesh.deltaCoeffs();
    mesh.nonOrthDeltaCoeffs();
    mesh.nonOrthCorrectionVectors();
}*/

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

    AD::registerInputVariable(alpha.begin(), alpha.end());

    //init_mesh(mesh);
    AD::switchTapeToPassive();

    const auto simpleFunc = [&]() -> bool {
        bool finished = !simple.loop();
        if(finished){
            Info << "finished" << endl;
            return true;
        }

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();

        // turbulence correct
        bool frozenTurbulence = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<bool>("frozenTurbulence",false);
        if(frozenTurbulence){
            Info << "\nWARNING: Calculating Adjoints with frozen Turbulence assumption\n" << endl;
            AD::switchTapeToPassive();
        }

        turbulence->correct();

        if(frozenTurbulence){
            AD::switchTapeToActive();
        }
        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;

        return false;
    };

    const auto costFunc = [&]() -> scalar {
        scalar J = CostFunction(mesh).eval();
        return J;
    };

    const auto writeFunc = [&]() -> void {
        forAll(alpha,i){
            sens[i] = AD::derivative(alpha[i]) / mesh.V()[i];
        }
        Info << "sensSum: " << gSum(sens) << endl;
        if(runTime.writeTime()){
            sens.write();
        }
    };

    simpleCheckpointer simpleCheck(runTime, simpleFunc, costFunc, writeFunc);
    simpleCheck.checkDB().addScalarCheckpoint(cumulativeContErr);
    simpleCheck.checkDB().addDictionaryCheckpoint(const_cast<Foam::dictionary&>(mesh.solverPerformanceDict()));

    simpleCheck.run();

    writeFunc();

    Info << "sens Sum: " << runTime.timeName() << "\t" << gSum(sens) << endl;

    //ADmode::tape_t::remove(ADmode::global_tape);

    Info << endl;
    Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    Info<< "End\n" << endl;
    return 0;
}


// ************************************************************************* //
