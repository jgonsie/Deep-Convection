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

#include "CheckInterface.H"
#include "CheckController.H"
#include "costFunctionLibrary.C"

std::vector<double> tapeMem;

class simpleCheckpointer : public CheckController{
private:
    // inherited from CheckController:
    // Foam::Time& runTime;
    // autoPtr<CheckInterface> checkInterface;

    // references for variables living in main
    Foam::fvMesh& mesh;
    Foam::simpleControl& simple;
    Foam::volScalarField& p;
    Foam::volVectorField& U;
    Foam::singlePhaseTransportModel& laminarTransport;
    autoPtr<incompressible::turbulenceModel>& turbulence;
    Foam::IOMRFZoneList& MRF;
    fv::options& fvOptions;
    volScalarField& alpha;
    volScalarField& sens;
    surfaceScalarField& phi;

    label&  pRefCell;
    scalar& pRefValue;
    scalar& cumulativeContErr;

    // member variables
    scalar J;
    bool frozenTurbulence;


public:
    simpleCheckpointer
    (
        Foam::Time& runTime,
        Foam::fvMesh& mesh,
        Foam::simpleControl& simple,
        Foam::volScalarField& p,
        Foam::volVectorField& U,
        Foam::singlePhaseTransportModel& laminarTransport,
        autoPtr<incompressible::turbulenceModel>& turbulence,
        IOMRFZoneList& MRF,
        fv::options& fvOptions,
        volScalarField& alpha,
        volScalarField& sens,
        surfaceScalarField& phi,
        label&  pRefCell,
        scalar& pRefValue,
        scalar& cumulativeContErr
    )
    :
        CheckController(runTime),
        mesh(mesh),
        simple(simple),
        p(p),
        U(U),
        laminarTransport(laminarTransport),
        turbulence(turbulence),
        MRF(MRF),
        fvOptions(fvOptions),
        alpha(alpha),
        sens(sens),
        phi(phi),
        pRefCell(pRefCell),
        pRefValue(pRefValue),
        cumulativeContErr(cumulativeContErr)
    {
        Info << "Avg Checkpoint Size: "
             << checkInterface().checkDatabase().getCheckDatabaseSize()
             << " MB" << endl;
        Info << "nCheckpoints: " << checkInterface().checkDatabase().nCheckpoints() << endl;

        //CheckObjectScalar* sObj = new CheckObjectScalar(cumulativeContErr);
        checkInterface().checkDatabase().addScalarCheckpoint(cumulativeContErr);
        checkInterface().checkDatabase().addDictionaryCheckpoint(const_cast<Foam::dictionary&>(mesh.solverPerformanceDict()));

        label tapeSizeMB = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<label>("tapeSizeMB",4096);
        Info << "Creating Tape, size: " << tapeSizeMB << endl;
        AD::createGlobalTape(tapeSizeMB/Pstream::nProcs());

        AD::registerInputVariable(alpha.begin(),alpha.end());

        AD::switchTapeToPassive();
        frozenTurbulence = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<bool>("frozenTurbulence",false);
        if(frozenTurbulence){
            Info << "\nWARNING: Calculating Adjoints with frozen Turbulence assumption\n" << endl;
        }
    }

    bool runStep(){
        static bool needsWrite = true;
        bool finished = !simple.loop();
        Info << "stopAt: " << runTime.stopAt() << " " << runTime.endTime() << endl;

        if(finished){
            needsWrite = false;
            return true;
        }

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity SIMPLE corrector
        {
            #include "UEqn.H"
            #include "pEqn.H"
        }

        laminarTransport.correct();
        bool wasActive = AD::isTapeActive();
        if(frozenTurbulence && wasActive)
            AD::switchTapeToPassive();

        turbulence->correct();

        if(frozenTurbulence && wasActive)
            AD::switchTapeToActive();

        if (needsWrite)
            runTime.write();

        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = "   << runTime.elapsedClockTime() << " s"
             << nl << endl;

        return false;
    }

    scalar calcCost(){
        Info << "ACTION::calcCost" << endl;
        return CostFunction(mesh).eval();
    }

    void postInterpret(){
        //Info << "Writing sensitivity for timestep " << runTime.timeName() << endl;
        for(int i = 0; i < alpha.size(); i++)
            sens[i] = AD::derivative(alpha[i]) / mesh.V()[i];
        Info << "sens Sum: " << runTime.timeName() << "\t" << gSum(sens) << endl;

        //runTime.stopAt(Foam::Time::stopAtControls::saEndTime);
        if(runTime.writeTime()){
            Info << "Post Interpret: Writing sensitivity for timestep " << runTime.timeName() << endl;
            sens.write();
        }
        //runTime.write();
    }
    void write(bool firstRun){
        if(firstRun)
            runTime.write();
        else{
            if(runTime.writeTime() || runTime.stopAt() == Foam::Time::stopAtControls::saWriteNow){
                Info << "Writing sensitivity for timestep " << runTime.timeName() << endl;
                for(int i = 0; i < alpha.size(); i++){
                    sens[i] = AD::derivative(alpha[i]) / mesh.V()[i];
                }
                sens.write();
            }
            //Info << "sens Sum: " << runTime.timeName() << "\t" << gSum(sens) << endl;
        }
    }

    void start(){

    }
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

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

    simpleCheckpointer simpleCheck
    (
         runTime,
         mesh,
         simple,
         p,
         U,
         laminarTransport,
         turbulence,
         MRF,
         fvOptions,
         alpha,
         sens,
         phi,
         pRefCell,
         pRefValue,
         cumulativeContErr
    );

    simpleCheck.run();

    //runTime.setTime(0,0);
    //sens.write();

    Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;

    Foam::profiling::print(Info);

    return 0;
}


// ************************************************************************* //
