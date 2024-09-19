/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  | Copyright (C) 2017 OpenCFD Ltd.
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
    chtMultiRegionSimpleFoam

Group
    grpHeatTransferSolvers

Description
    Steady-state solver for buoyant, turbulent fluid flow and solid heat
    conduction with conjugate heat transfer between solid and fluid regions.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "rhoThermo.H"
#include "turbulentFluidThermoModel.H"
#include "fixedGradientFvPatchFields.H"
#include "regionProperties.H"
#include "solidThermo.H"
#include "radiationModel.H"
#include "fvOptions.H"
#include "coordinateSystem.H"
#include "loopControl.H"

#include "CheckController.H"
#include "CheckInterface.H"
#include "CheckDict.H"
//#include "CheckObject.H"
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "CheckController.H"
//#include "costFunctionLibrary.C"

//#include <functional>

void init_mesh(Foam::fvMesh& mesh){
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
        "Steady-state solver for buoyant, turbulent fluid flow and solid heat"
        " conduction with conjugate heat transfer"
        " between solid and fluid regions."
    );

    #define NO_CONTROL
    #define CREATE_MESH createMeshesPostProcess.H
    #include "postProcess.H"

    #include "setRootCaseLists.H"
    #include "createTime.H"
    //#include "createMeshes.H"

    regionProperties rp(runTime);
    const wordList solidsNames(rp["solid"]);
    const wordList fluidNames(rp["fluid"]);
    PtrList<fvMesh> solidRegions(solidsNames.size());
    PtrList<fvMesh> fluidRegions(fluidNames.size());

    forAll(solidsNames, i)
    {
        Info << "Create solid mesh for region " << solidsNames[i]
             << " for time = " << runTime.timeName() << nl << endl;

        solidRegions.set
        (
            i,
            new fvMesh
            (
                IOobject
                (
                    solidsNames[i],
                    runTime.timeName(),
                    runTime,
                    IOobject::MUST_READ
                )
            )
        );
    }
    forAll(fluidNames, i)
    {
        Info << "Create fluid mesh for region " << fluidNames[i]
             << " for time = " << runTime.timeName() << nl << endl;

        fluidRegions.set
        (
            i,
            new fvMesh
            (
                IOobject
                (
                    fluidNames[i],
                    runTime.timeName(),
                    runTime,
                    IOobject::MUST_READ
                )
            )
        );
    }

    #include "createFields.H"

    label tapeSizeMB = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<label>("tapeSizeMB",4096);
    Info << "Creating Tape, size: " << tapeSizeMB << endl;
    AD::createGlobalTape(tapeSizeMB/Pstream::nProcs());

    AD::switchTapeToActive();

    forAll(solidRegions,i){
        init_mesh(solidRegions[i]);
    }
    forAll(fluidRegions,i){
        init_mesh(fluidRegions[i]);
    }

    CheckInterface check(runTime);
    CheckDict checkDict(runTime);
    CheckDatabase checkDB(runTime,checkDict);

    forAll(alphaFluid,i){
        AD::registerInputVariable(alphaFluid[i].begin(),alphaFluid[i].end());
    }
    AD::position_t reset_to = AD::getTapePosition();

    //AD::switch_tape_to_passive();

    //#include "initContinuityErrs.H"
    scalar cumulativeContErr = 0;
    bool firstStep = true;

    scalar cost1 = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("cost1",1.0);
    scalar cost2 = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("cost2",1.0);

    scalar logistic_k = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("logistic_k",2.0);
    scalar logistic_shift = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("logistic_shift",5.0);
    label nOptSteps = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<label>("nOptSteps",100);
    scalar optEpsilon = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("optTolerance",5e-2);
    scalar alphaMax = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("alphaMax",1000);
    scalar optStepWidth = fluidRegions[0].solutionDict().subDict("SIMPLE").lookupOrDefault<scalar>("optStepwidth",0.1);

    while(runTime.loop()){
        label optZoneID = fluidRegions[0].cellZones().findZoneID("optZone");
        if(optZoneID >= 0 && (runTime.timeIndex()+1) % 40 == 0){
            cellZone& optZone = fluidRegions[0].cellZones()[optZoneID];
            scalarMinMax alpha_range(0, 100);
            Info << "OPT" << endl;
            for(auto cI : optZone){
                AD::value(alphaFluid[0][cI]) = std::min(10.0,AD::value(alphaFluid[0][cI])-AD::passiveValue(optStepWidth*sensFluid[0][cI]));
            }
        }

        checkDB.registerAdjoints(); // store tape indices

        Info << "Time = " << runTime.timeName() << nl << endl;

        forAll(fluidRegions, i)
        {
            Info << "\nSolving for fluid region " << fluidRegions[i].name() << endl;
            #include "setRegionFluidFields.H"
            #include "readFluidMultiRegionSIMPLEControls.H"
            #include "solveFluid.H"
        }

        forAll(solidRegions, i)
        {
            Info << "\nSolving for solid region " << solidRegions[i].name() << endl;
            #include "setRegionSolidFields.H"
            #include "readSolidMultiRegionSIMPLEControls.H"
            #include "solveSolid.H"
        }

        // Additional loops for energy solution only
        {
            loopControl looping(runTime, "SIMPLE", "energyCoupling");

            while (looping.loop())
            {
                Info << nl << looping << nl;

                forAll(fluidRegions, i)
                {
                    Info << "\nSolving for fluid region " << fluidRegions[i].name() << endl;
                   #include "setRegionFluidFields.H"
                   #include "readFluidMultiRegionSIMPLEControls.H"
                   frozenFlow = true;
                   #include "solveFluid.H"
                }

                forAll(solidRegions, i)
                {
                    Info << "\nSolving for solid region " << solidRegions[i].name() << endl;
                    #include "setRegionSolidFields.H"
                    #include "readSolidMultiRegionSIMPLEControls.H"
                    #include "solveSolid.H"
                }
            }
        }
        runTime.printExecutionTime(Info);

        scalar Tavg_outlet = gAverage(thermoFluid[0].T().boundaryField()[1]);
        scalar Tavg_solid = gAverage(thermos[0].T());
        scalar pAvg_inlet = gAverage(p_rghFluid[0].boundaryField()[0]);
        Info << "Average p_inlet: "  << fluidRegions[0].boundary()[0].name() << " " << pAvg_inlet << endl;
        Info << "Average T_outlet: "  << fluidRegions[0].boundary()[0].name() << " " << Tavg_outlet << endl;
        //Info << "Average T_solid: "  <<  Tavg_outlet << endl;

        scalar J = cost1*Tavg_outlet + cost2*pAvg_inlet;

        Info << "cost " << pAvg_inlet << " " << Tavg_outlet << " " << Tavg_solid << " " << gSum(alphaFluid[0]) << " " << J << " " << gMin(sensFluid[0]) << endl;

        AD::derivative(J)=1.0;

        if(!firstStep){
            checkDB.restoreAdjoints();
        }else{
            firstStep = false;
        }

        AD::interpretTape(); //_to(interpret_to);
        checkDB.storeAdjoints();

        forAll(alphaFluid,i){
            forAll(alphaFluid[i],j){
                sensFluid[i][j] = AD::derivative(alphaFluid[i][j]) / fluidRegions [i].V()[j];;
            }
            /*if(runTime.writeTime()){
                sensFluid[i].write();
            }*/
            Info << "sensSum: " << gSum(sensFluid[i]) << endl;
        }

        runTime.write();

        AD::resetTapeTo(reset_to);
        AD::zeroAdjointVector();
    }

    Info << "End\n" << endl;

    return 0;
}

// ************************************************************************* //
