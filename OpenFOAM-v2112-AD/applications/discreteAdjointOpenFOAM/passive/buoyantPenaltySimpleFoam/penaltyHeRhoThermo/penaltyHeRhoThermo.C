/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2015-2017 OpenCFD Ltd.
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

#include "penaltyHeRhoThermo.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
void Foam::penaltyHeRhoThermo<BasicPsiThermo, MixtureType>::calculate
(
    const volScalarField& p,
    volScalarField& T,
    volScalarField& he,
    volScalarField& psi,
    volScalarField& rho,
    volScalarField& mu,
    volScalarField& alpha,
    const bool doOldTimes
)
{
    // Note: update oldTimes before current time so that if T.oldTime() is
    // created from T, it starts from the unconverted T
    if (doOldTimes && (p.nOldTimes() || T.nOldTimes()))
    {
        calculate
        (
            p.oldTime(),
            T.oldTime(),
            he.oldTime(),
            psi.oldTime(),
            rho.oldTime(),
            mu.oldTime(),
            alpha.oldTime(),
            true
        );
    }

    const scalarField& hCells = he.primitiveField();
    const scalarField& pCells = p.primitiveField();

    scalarField& TCells = T.primitiveFieldRef();
    //scalarField& psiCells = psi.primitiveFieldRef();
    scalarField& rhoCells = rho.primitiveFieldRef();
    scalarField& muCells = mu.primitiveFieldRef();
    scalarField& alphaCells = alpha.primitiveFieldRef();
    
    scalar fluidCp = this->mixtureDict_.subDict("mixture").subDict("thermodynamics").getScalar("Cp");
    scalar fluidMu = this->mixtureDict_.subDict("mixture").subDict("transport").getScalar("mu");
    scalar fluidPr = this->mixtureDict_.subDict("mixture").subDict("transport").getScalar("Pr");
    scalar fluidRho = this->mixtureDict_.subDict("mixture").subDict("equationOfState").getScalar("rho");

    scalar solidRho = this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("rho");
    scalar solidKappa = this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("kappa");
    scalar solidCp = this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("Cp");
    scalar alphaSolid = 8954*11.234e5;
    Info << "alpha dim " << alpha.dimensions() << endl;
    
    const scalar Tref = 298.15;
    const scalar f = 100;

    forAll(TCells, celli)
    {
        const typename MixtureType::thermoType& mixture_ =
            this->cellMixture(celli);
        //const typename MixtureType::transportType& transport_ ;

        if (this->updateT())
        {
            TCells[celli] = mixture_.THE
            (
                hCells[celli],
                pCells[celli],
                TCells[celli]
            );
        }

        // psiCells[celli] = mixture_.psi(pCells[celli], TCells[celli]);
        rhoCells[celli] = fluidRho; //mixture_.rho(pCells[celli], TCells[celli]);

        muCells[celli] = fluidMu; //mixture_.mu(pCells[celli], TCells[celli]);
        // alpha = mu/Pr = kappa/Cp
        alphaCells[celli] = penalty_[celli] * alphaSolid + scalar(1.0-penalty_[celli]) * fluidMu / fluidPr; //mixture_.alphah(pCells[celli], TCells[celli]); /*penalty_[celli]*scalar(solidKappa_/solidCp_) + scalar(1.0-penalty_[celli]) * mixture_.alphah(pCells[celli], TCells[celli]);*/
    }

    const volScalarField::Boundary& pBf = p.boundaryField();
    volScalarField::Boundary& TBf = T.boundaryFieldRef();
    volScalarField::Boundary& psiBf = psi.boundaryFieldRef();
    volScalarField::Boundary& rhoBf = rho.boundaryFieldRef();
    volScalarField::Boundary& heBf = he.boundaryFieldRef();
    volScalarField::Boundary& muBf = mu.boundaryFieldRef();
    volScalarField::Boundary& alphaBf = alpha.boundaryFieldRef();
    const volScalarField::Boundary& penaltyBf = penalty_.boundaryField();



    forAll(pBf, patchi)
    {
        const fvPatchScalarField& pp = pBf[patchi];
        fvPatchScalarField& pT = TBf[patchi];
        fvPatchScalarField& ppsi = psiBf[patchi];
        fvPatchScalarField& prho = rhoBf[patchi];
        fvPatchScalarField& phe = heBf[patchi];
        fvPatchScalarField& pmu = muBf[patchi];
        fvPatchScalarField& palpha = alphaBf[patchi];
        const fvPatchScalarField& ppenalty = penaltyBf[patchi];

        if (pT.fixesValue())
        {
            forAll(pT, facei)
            {
                const typename MixtureType::thermoType& mixture_ =
                    this->patchFaceMixture(patchi, facei);

                phe[facei] = (ppenalty[facei]*alphaSolid + (1-ppenalty[facei])*fluidCp) * (pT[facei] - Tref); //mixture_.HE(pp[facei], pT[facei]);

                //ppsi[facei] = mixture_.psi(pp[facei], pT[facei]);
                prho[facei] = fluidRho; //mixture_.rho(pp[facei], pT[facei]);
                pmu[facei] = fluidMu; //mixture_.mu(pp[facei], pT[facei]);
                palpha[facei] = ppenalty[facei]*alphaSolid + scalar(1.0-ppenalty[facei]) * fluidMu / fluidPr; // solidKappa / solidCp
                //palpha[facei] = fluidMu / fluidPr; //penalty_[facei]*solidKappa_/solidCp_ + scalar(1.0-penalty_[facei])*mixture_.alphah(pp[facei], pT[facei]);
            }
        }
        else
        {
            forAll(pT, facei)
            {
                const typename MixtureType::thermoType& mixture_ =
                    this->patchFaceMixture(patchi, facei);

                if (this->updateT())
                {
                    pT[facei] = mixture_.THE(phe[facei], pp[facei], pT[facei]);
                }

                //ppsi[facei] = mixture_.psi(pp[facei], pT[facei]);
                prho[facei] = fluidRho; //mixture_.rho(pp[facei], pT[facei]);
                pmu[facei] = fluidMu; //mixture_.mu(pp[facei], pT[facei]);
                palpha[facei] = ppenalty[facei]*alphaSolid + scalar(1.0-ppenalty[facei]) * fluidMu / fluidPr; //penalty_[facei]*solidKappa_/solidCp_ + scalar(1.0-penalty_[facei])*mixture_.alphah(pp[facei], pT[facei]);
            }
        }
    }
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::penaltyHeRhoThermo<BasicPsiThermo, MixtureType>::penaltyHeRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName
)
:
    heThermo<BasicPsiThermo, MixtureType>(mesh, phaseName),
    penalty_(mesh.lookupObject<volScalarField>("penalty")),
    solidRho_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("rho")),
    solidCp_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("Cp")),
    solidKappa_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("kappa"))
{
    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        true                    // Create old time fields
    );
}


template<class BasicPsiThermo, class MixtureType>
Foam::penaltyHeRhoThermo<BasicPsiThermo, MixtureType>::penaltyHeRhoThermo
(
    const fvMesh& mesh,
    const word& phaseName,
    const word& dictName
)
:
    heThermo<BasicPsiThermo, MixtureType>(mesh, phaseName, dictName),
    penalty_(mesh.lookupObject<volScalarField>("penalty")),
    solidRho_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("rho")),
    solidCp_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("Cp")),
    solidKappa_(this->mixtureDict_.subDict("mixture").subDict("solidProperties").getScalar("kappa"))
{
    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        true                    // Create old time fields
    );
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
Foam::penaltyHeRhoThermo<BasicPsiThermo, MixtureType>::~penaltyHeRhoThermo()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class BasicPsiThermo, class MixtureType>
void Foam::penaltyHeRhoThermo<BasicPsiThermo, MixtureType>::correct()
{
    DebugInFunction << endl;

    calculate
    (
        this->p_,
        this->T_,
        this->he_,
        this->psi_,
        this->rho_,
        this->mu_,
        this->alpha_,
        false           // No need to update old times
    );

    DebugInFunction << "Finished" << endl;
}

// ************************************************************************* //
