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

#include <functional>

class chtCheckpointer : public CheckController{
private:
    std::function<bool()> runStepLambda;
    std::function<scalar()> calcCostLambda;
    std::function<void()> writeLambda;
public:
    chtCheckpointer
    (
        Foam::Time& runTime,
        const std::function<bool()>& runStepLambda,
        const std::function<scalar()>& calcCostLambda,
        const std::function<void()>& writeLambda
    )
    :
        CheckController(runTime),
        runStepLambda(runStepLambda),
        calcCostLambda(calcCostLambda),
        writeLambda(writeLambda)
    {
    }

    bool runStep(){
        return runStepLambda();
    }

    scalar calcCost(){
        return calcCostLambda();
    }

    void postInterpret(){}
    void write(bool firstRun){writeLambda();}
    void start(){}

    auto& checkDB(){
        return checkInterface().checkDatabase();
    }
};

// force initialization of mesh while tape is switched on
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

void calcShapeSens(volScalarField& shapeSens){
    const fvMesh& mesh = shapeSens.mesh();
    const Foam::wordList sensOutputPatches(mesh.solutionDict().subDict("SIMPLE").lookup("sensOutputPatches"));

    // how often is each point referenced in different faces?
    std::vector<label> pointRefs(mesh.points().size());
    forAll(sensOutputPatches,cI)
    {
        Foam::label bi = mesh.boundaryMesh().findPatchID(sensOutputPatches[cI]);
        forAll(mesh.boundary()[bi],i){
            const labelList face_points = mesh.boundary()[bi].patch()[i];
            for(const label& labelI : face_points){
                pointRefs[labelI]++;
            }
        }
    }

    forAll(sensOutputPatches,cI)
    {
        Info << "### Evaluating sensOutputPatch = " << sensOutputPatches[cI] << endl;
        Foam::label bi = mesh.boundaryMesh().findPatchID(sensOutputPatches[cI]);
        forAll(mesh.boundary()[bi],i){
            // loop over all faces in boundary bi
            // list of point indices for face i on bi
            const labelList face_points = mesh.boundary()[bi].patch()[i];
            // cell in domain corresponding to boundary face
            const label face_cell = mesh.boundary()[bi].faceCells()[i];
            const Foam::vector face_normal = mesh.boundary()[bi].nf()()[i];
            Foam::vector sensVec(0.0,0.0,0.0);
            forAll(face_points,fp){
                const point& pt = mesh.points()[face_points[fp]];
                sensVec[0] += AD::derivative(pt[0]); /// face_points.size();
                sensVec[1] += AD::derivative(pt[1]); /// face_points.size();
                sensVec[2] += AD::derivative(pt[2]); /// face_points.size();
            }
            sensVec /= (face_points.size());
            // scalar product of sensitivity vector with face normal
            shapeSens[face_cell] = (sensVec & face_normal) / mesh.boundary()[bi].magSf()[i];
            //surfSens.boundaryFieldRef()[bi][i] = sens[face_cell];
        }
    }
}

/*void zero_cells(volScalarField& sens){
    const fvMesh& mesh = sens.mesh();
    const Foam::wordList zeroPatches = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<Foam::wordList>("zeroPatches",Foam::wordList(0));

    // zero sensitivities at inlets / outlets
    std::set<label> zeroCells;
    forAll(zeroPatches,cI)
    {
        Foam::label bi = mesh.boundaryMesh().findPatchID(zeroPatches[cI]);
        forAll(mesh.boundary()[bi],i){
            zeroCells.insert(mesh.boundary()[bi].faceCells()[i]);
        }
    }
    if(zeroCells.size()>0){
        auto newCells = zeroCells;
        for(int i = 0; i < 4; i++){
            auto tmpCells = newCells;
            newCells.clear();
            for(auto it = tmpCells.begin(); it != tmpCells.end(); it++){
                auto neigh = mesh.cellCells(*it);
                forAll(neigh,j){
                    auto ret = zeroCells.insert(neigh[j]);
                    if(ret.second){ // only add new elements for further exploration
                        newCells.insert(neigh[j]);
                    }
                }
            }
        }
        //for(auto it = zeroCells.begin(); it != zeroCells.end(); it++){
        for(const label c : zeroCells){
            sens[c] = 0;
        }
    }
}*/

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

    AD::createGlobalTape(); // allocate tape before mesh generation
    AD::switchTapeToActive(); // inputs are registered in primitiveMesh classes

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

    forAll(solidRegions,i){
        init_mesh(solidRegions[i]);
    }
    forAll(fluidRegions,i){
        init_mesh(fluidRegions[i]);
    }

    AD::switchTapeToPassive();

    //#include "initContinuityErrs.H"
    scalar cumulativeContErr = 0;

    std::function<bool()> runStep = [&]() -> bool {
        static bool needsWrite = true;
        bool finished = !runTime.loop();
        if(finished){
            needsWrite = false;
            return true;
        }

        label optZoneID = fluidRegions[0].cellZones().findZoneID("optZone");
        /*if(optZoneID >= 0 && (runTime.timeIndex() % 10) == 0){
            cellZone& optZone = fluidRegions[0].cellZones()[optZoneID];
            scalarMinMax alpha_range(0, 100);
            Info << "OPT" << endl;
            for(auto cI : optZone){
                AD::value(alphaFluid[0][cI]) = AD::passiveValue(alpha_range.clip(alphaFluid[0][cI] - 0.1*sensFluid[0][cI]));
            }
        }*/

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
        if(needsWrite){
            runTime.write();
        }

        runTime.printExecutionTime(Info);

        return false;
    };

    std::function<scalar()> calcCost = [&]() -> scalar {
        scalar Tavg_outlet = gAverage(thermoFluid[0].T().boundaryField()[1]);
        scalar Tavg_solid = gAverage(thermos[0].T());
        scalar pAvg_inlet = gAverage(p_rghFluid[0].boundaryField()[0]);
        //scalar Tavg_solid = 0.5*(gAverage(thermos[0].T().boundaryField()[0]) + gAverage(thermos[0].T().boundaryField()[1]));
        //scalar Tavg_solid = gAverage(thermos[0].T().boundaryField()[0]);

        //Info << "Average T_outlet: " << fluidRegions[0].boundary()[1].name() << " " << Tavg_outlet << endl;
        //Info << "Average p_inlet: "  << fluidRegions[0].boundary()[0].name() << " " << pAvg_inlet << endl;
        //Info << "Average T_solid: "  << solidRegions[0].boundary()[0].name() << " " << Tavg_solid << endl;
        //scalar J = -Tavg_outlet + 1000*pAvg_inlet;
        //scalar J = -Tavg_outlet + 100*pAvg_inlet;
        scalar J = -Tavg_outlet + pAvg_inlet;

        //Info << "cost " << pAvg_inlet << " " << Tavg_outlet << " " << Tavg_solid << " " << gSum(alphaFluid[0]) << " " << J << " " << gMin(sensFluid[0]) << endl;
        return J;
    };

    std::function<void()> write = [&]{
        /*forAll(alphaFluid,i){
            forAll(alphaFluid[i],j){
                sensFluid[i][j] = AD::derivative(alphaFluid[i][j]) / fluidRegions [i].V()[j];;
            }
            Info << "sensSum: " << gSum(sensFluid[i]) << endl;
        }*/
        if(runTime.writeTime()){
            forAll(alphaFluid,i){
                calcShapeSens(shapeSensFluid[i]);
                shapeSensFluid[i].write();
            }
        }
    };

    chtCheckpointer chtCheck
    (
        runTime,
        runStep,
        calcCost,
        write
    );
    chtCheck.checkDB().addScalarCheckpoint(cumulativeContErr);

    chtCheck.run();
    
    ////ADmode::global_tape->switch_to_passive();

    runTime.setTime(0,label(0));
    AD::interpretTape();
    forAll(alphaFluid,i){
        calcShapeSens(shapeSensFluid[i]);
        shapeSensFluid[i].write();
    }

    //Info << "sensSum: " << gSum(sensFluid[0]) << endl;
    Info << "End\n" << endl;

    return 0;
}

// ************************************************************************* //
