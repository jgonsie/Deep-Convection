/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2013-2017 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/

#include "eddyViscosityRAM.H"
#include "fvc.H"
#include "fvm.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
Foam::eddyViscosityRAM<BasicTurbulenceModel>::eddyViscosityRAM
(
    const word& type,
    const alphaField& alpha,
    const rhoField& rho,
    const volVectorField& U,
    const surfaceScalarField& alphaRhoPhi,
    const surfaceScalarField& phi,
    const transportModel& transport,
    const word& propertiesName
)
:
    linearViscousStress<BasicTurbulenceModel>
    (
        type,
        alpha,
        rho,
        U,
        alphaRhoPhi,
        phi,
        transport,
        propertiesName
    ),
    nut_
    (
        IOobject
        (
            IOobject::groupName("nut", alphaRhoPhi.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        dimensionedScalar("nut", dimensionSet(0,2,-1,0,0,0,0), 0) //I initialize with all zeros. Then in the kRAM/OmegaRAM functions I delete this initialization
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicTurbulenceModel>
bool Foam::eddyViscosityRAM<BasicTurbulenceModel>::read()
{
    return BasicTurbulenceModel::read();
}


template<class BasicTurbulenceModel>
Foam::tmp<Foam::volSymmTensorField>
Foam::eddyViscosityRAM<BasicTurbulenceModel>::R() const
{
    tmp<volScalarField> tk(k());

    // Get list of patchField type names from k
    wordList patchFieldTypes(tk().boundaryField().types());

    // For k patchField types which do not have an equivalent for symmTensor
    // set to calculated
    forAll(patchFieldTypes, i)
    {
        if
        (
           !fvPatchField<symmTensor>::patchConstructorTablePtr_
                ->found(patchFieldTypes[i])
        )
        {
            patchFieldTypes[i] = calculatedFvPatchField<symmTensor>::typeName;
        }
    }

    return tmp<volSymmTensorField>
    (
        new volSymmTensorField
        (
            IOobject
            (
                IOobject::groupName("R", this->alphaRhoPhi_.group()),
                this->runTime_.timeName(),
                this->mesh_,
                IOobject::NO_READ,
                IOobject::NO_WRITE,
                false
            ),
            ((2.0/3.0)*I)*tk() - (nut_)*dev(twoSymm(fvc::grad(this->U_))),
            patchFieldTypes
        )
    );
}


template<class BasicTurbulenceModel>
void Foam::eddyViscosityRAM<BasicTurbulenceModel>::validate()
{
    correctNut();
}


template<class BasicTurbulenceModel>
void Foam::eddyViscosityRAM<BasicTurbulenceModel>::correct()
{
    BasicTurbulenceModel::correct();
}

//- Reads the turbulence kinetic energy dissipation rate from RAM
template<class BasicTurbulenceModel>
void Foam::eddyViscosityRAM<BasicTurbulenceModel>::nutRAM(dictionary nut_dict)
{
    Info<< "Esto es nutRAM "<< endl;
    nut_.~GeometricField();
    new (&nut_) volScalarField
    (
        IOobject
        (
            IOobject::groupName("nut", this->alphaRhoPhi_.group()),
            this->runTime_.timeName(),
            this->mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        this->mesh_,
        nut_dict        
    );
}

// ************************************************************************* //
