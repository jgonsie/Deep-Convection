#include "costFunctionLibrary.H"
#include "fvCFD.H"
// #include "forces.H"
#include "turbulentTransportModel.H"
#include "fvcGrad.H"

CostFunction::CostFunction(Foam::fvMesh &mesh)
    :
      mesh(mesh)
{}

Foam::scalar CostFunction::eval()
{
    //const Foam::wordList costFunctionPatches =
    //        mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<Foam::wordList>("costFunctionPatches",Foam::wordList(0));
    const Foam::wordList costFunctionPatches(
            mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionPatches")
        );
    const Foam::word costFunction =
            mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<Foam::word>("costFunction","");

    Foam::scalar J = 0;
    Foam::Info << "Cost Function: " << costFunction << " evaluated on: " << costFunctionPatches << Foam::endl;
    if(costFunction=="pressureLoss"){
        J = eval_pressure(costFunctionPatches);
    }
    else if(costFunction=="drag" || costFunction=="lift"){
        J = eval_drag(costFunctionPatches);
    }
    else if(costFunction=="drag_legacy"){
        J = eval_drag_legacy(costFunctionPatches);
    }
    else if(costFunction=="liftDragRatio"){
        J = eval_liftdragratio(costFunctionPatches);
    }
    else{
        Foam::Info << "Unknown Cost Function!" << Foam::endl;
        Foam::Info << "Options are: pressureLoss, drag, lift, liftDragRatio" << Foam::endl;
    }
    Info<< "cost: " << J << endl;
    return J;
}

Foam::scalar CostFunction::eval_pressure(const Foam::wordList &costFunctionPatches)
{
    const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");
    const Foam::surfaceScalarField& phi = mesh.lookupObject<Foam::surfaceScalarField>("phi");

    scalar J = 0;
    forAll(costFunctionPatches,cI)
    {
        Foam::label patchI = mesh.boundaryMesh().findPatchID(costFunctionPatches[cI] );
        const Foam::fvPatch& patch = mesh.boundary()[patchI];
        J += gSum
                (
                    - phi.boundaryField()[patchI]*(p.boundaryField()[patchI]
                    + 0.5*magSqr(phi.boundaryField()[patchI]/patch.magSf()))
                );
    }
    return J;
}

Foam::scalar CostFunction::eval_drag_legacy(const Foam::wordList &costFunctionPatches)
{
    const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");
    //const Foam::surfaceScalarField& phi = mesh.lookupObject<Foam::surfaceScalarField>("phi");

    //const Foam::vector direction = mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<Foam::vector>("costFunctionDirection",Foam::vector::zero);
    const Foam::vector direction = Foam::vector(
            mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionDirection")
        );
    Foam::Info << "Direction: " << direction << Foam::endl;
    scalar J = 0;
    // calculate lift/drag
    forAll(costFunctionPatches,cI)
    {
        label patchI = mesh.boundaryMesh().findPatchID( costFunctionPatches[cI] );
        const fvPatch& patch = mesh.boundary()[patchI];
        // integrate over p * surface normals, x-component for drag, y-component for lift
        J += direction & gSum( p.boundaryField()[patchI]*patch.Sf() ); // scalar product
    }
    return J;
}

Foam::scalar CostFunction::eval_drag(const Foam::wordList &costFunctionPatches)
{
    const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");

    const Foam::vector direction = Foam::vector(
        mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionDirection")
    );
    Foam::Info << "Direction: " << direction << Foam::endl;


    const bool includeViscous(
        mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<bool>("includeViscous",true)
    );
    Info << "Selecting includeViscous = " << includeViscous << endl;

    scalar J = 0;
    // calculate lift/drag
    forAll(costFunctionPatches,cI)
    {
        label patchI = mesh.boundaryMesh().findPatchID( costFunctionPatches[cI] );
        const fvPatch& patch = mesh.boundary()[patchI];

        tmp<volSymmTensorField> tdevRhoReff = devRhoReff();
        const volSymmTensorField::Boundary& devRhoReffb
            = tdevRhoReff().boundaryField();

        // Normal force (pressure)
        vectorField fN(patch.Sf()*p.boundaryField()[patchI]);
        // Tangential force (viscous)
        vectorField fT(patch.Sf() & devRhoReffb[patchI]);

        vector totalForce(0,0,0);

        if(includeViscous) {
            totalForce = gSum(fN+fT);
        }
        else {
            totalForce = gSum(fN);
        }

        // integrate over p * surface normals, x-component for drag, y-component for lift
        // J += direction & gSum( p.boundaryField()[patchI]*patch.Sf() ); // scalar product
        J += direction & totalForce; // scalar product
    }
    return J;
}

Foam::scalar CostFunction::eval_liftdragratio(const Foam::wordList &costFunctionPatches)
{
    const Foam::volScalarField& p = mesh.lookupObject<Foam::volScalarField>("p");

    const Foam::vector liftdirection = Foam::vector(
            mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionLiftDirection")
        );
    const Foam::vector dragdirection = Foam::vector(
            mesh.solutionDict().subDict("SIMPLE").lookup("costFunctionDragDirection")
        );
    Foam::Info << "Direction: " << liftdirection << " (lift) " << dragdirection << " (drag) " << Foam::endl;

    const bool includeViscous(
        mesh.solutionDict().subDict("SIMPLE").lookupOrDefault<bool>("includeViscous",true)
    );
    Info << "Selecting includeViscous = " << includeViscous << endl;

    scalar J = 0;
    // calculate lift/drag
    forAll(costFunctionPatches,cI)
    {
        label patchI = mesh.boundaryMesh().findPatchID( costFunctionPatches[cI] );
        const fvPatch& patch = mesh.boundary()[patchI];

        tmp<volSymmTensorField> tdevRhoReff = devRhoReff();
        const volSymmTensorField::Boundary& devRhoReffb
            = tdevRhoReff().boundaryField();

        // Normal force (pressure)
        vectorField fN(patch.Sf()*p.boundaryField()[patchI]);
        // Tangential force (viscous)
        vectorField fT(patch.Sf() & devRhoReffb[patchI]);
        
        vector totalForce(0,0,0);

        if(includeViscous) {
            totalForce = gSum(fN+fT);
        }
        else {
            totalForce = gSum(fN);
        }

        // integrate over p * surface normals, x-component for drag, y-component for lift
        // J += (liftdirection & gSum(p.boundaryField()[patchI]*patch.Sf()))/(dragdirection & gSum( p.boundaryField()[patchI]*patch.Sf() )); // scalar product
        J += (liftdirection & totalForce) / (dragdirection & totalForce);
    }
    return J;
}

// lm: adapted from src/functionObjects/forces/forces/forces.C
// code for incompressible was removed, and rho=1 constant (so far)
Foam::tmp<Foam::volSymmTensorField> CostFunction::devRhoReff() const
{
    typedef Foam::incompressible::turbulenceModel icoTurbModel;


    if (mesh.foundObject<icoTurbModel>(icoTurbModel::propertiesName))
    {
        const Foam::incompressible::turbulenceModel& turb =
            mesh.lookupObject<icoTurbModel>(icoTurbModel::propertiesName);

        Foam::Info << "calculating devRhoReff using turb.devReff() " << icoTurbModel::propertiesName << endl;

        return rho()*turb.devReff();
    }
    else
    if (mesh.foundObject<transportModel>("transportProperties"))
    {
        const Foam::transportModel& laminarT =
            mesh.lookupObject<Foam::transportModel>("transportProperties");

        const Foam::volVectorField& U = mesh.lookupObject<Foam::volVectorField>("U");

        Foam::Info << "calculating devRhoReff using laminarT.nu()" << endl;

        return scalar(-rho())*laminarT.nu()*Foam::dev(Foam::twoSymm(Foam::fvc::grad(U)));
    }
    else if (mesh.foundObject<dictionary>("transportProperties"))
    {
        const Foam::dictionary& transportProperties =
            mesh.lookupObject<Foam::dictionary>("transportProperties");

        Foam::dimensionedScalar nu(transportProperties.lookup("nu"));

        const Foam::volVectorField& U = mesh.lookupObject<Foam::volVectorField>("U");

        Foam::Info << "calculating devRhoReff using nu" << endl;

        return scalar(-rho())*nu*Foam::dev(Foam::twoSymm(Foam::fvc::grad(U)));
    }
    else
    {
        FatalErrorInFunction
            << "No valid model for viscous stress calculation"
            << exit(FatalError);

        return volSymmTensorField::null();
    }
}

// lm: TODO so far hard-coded...
Foam::scalar CostFunction::rho() const
{
    return 1.0;
}
