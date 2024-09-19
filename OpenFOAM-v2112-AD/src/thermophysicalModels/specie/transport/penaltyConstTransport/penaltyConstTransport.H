/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2020 OpenCFD Ltd.
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

Class
    Foam::penaltyConstTransport

Group
    grpSpecieTransport

Description
    Constant properties Transport package.
    Templated into a given thermodynamics package (needed for thermal
    conductivity).

SourceFiles
    penaltyConstTransportI.H
    penaltyConstTransport.C

\*---------------------------------------------------------------------------*/

#ifndef penaltyConstTransport_H
#define penaltyConstTransport_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward Declarations

template<class Thermo> class penaltyConstTransport;

template<class Thermo>
inline penaltyConstTransport<Thermo> operator+
(
    const penaltyConstTransport<Thermo>&,
    const penaltyConstTransport<Thermo>&
);

template<class Thermo>
inline penaltyConstTransport<Thermo> operator*
(
    const scalar,
    const penaltyConstTransport<Thermo>&
);

template<class Thermo>
Ostream& operator<<
(
    Ostream&,
    const penaltyConstTransport<Thermo>&
);


/*---------------------------------------------------------------------------*\
                           Class penaltyConstTransport Declaration
\*---------------------------------------------------------------------------*/

template<class Thermo>
class penaltyConstTransport
:
    public Thermo
{
    // Private Data

        //- Constant dynamic viscosity [Pa.s]
        scalar mu_;

        //- Reciprocal Prandtl Number [-]
        scalar rPr_;


    // Private Member Functions

        //- Construct from components
        inline penaltyConstTransport
        (
            const Thermo& t,
            const scalar mu,
            const scalar Pr
        );


public:

    // Generated Methods: copy construct, copy assignment


    // Constructors

        //- Construct as named copy
        inline penaltyConstTransport(const word&, const penaltyConstTransport&);

        //- Construct from dictionary
        explicit penaltyConstTransport(const dictionary& dict);

        //- Construct and return a clone
        inline autoPtr<penaltyConstTransport> clone() const;

        // Selector from dictionary
        inline static autoPtr<penaltyConstTransport> New(const dictionary& dict);


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "const<" + Thermo::typeName() + '>';
        }

        //- Dynamic viscosity [kg/ms]
        inline scalar mu(const scalar p, const scalar T) const;

        //- Thermal conductivity [W/mK]
        inline scalar kappa(const scalar p, const scalar T) const;

        //- Thermal diffusivity of enthalpy [kg/ms]
        inline scalar alphah(const scalar p, const scalar T) const;

        // Species diffusivity
        //inline scalar D(const scalar p, const scalar T) const;

        //- Write to Ostream
        void write(Ostream& os) const;


    // Member Operators

        inline void operator+=(const penaltyConstTransport&);

        inline void operator*=(const scalar);


    // Friend Operators

        friend penaltyConstTransport operator+ <Thermo>
        (
            const penaltyConstTransport&,
            const penaltyConstTransport&
        );

        friend penaltyConstTransport operator* <Thermo>
        (
            const scalar,
            const penaltyConstTransport&
        );


    // IOstream Operators

        friend Ostream& operator<< <Thermo>
        (
            Ostream&,
            const penaltyConstTransport&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "penaltyConstTransportI.H"

#ifdef NoRepository
    #include "penaltyConstTransport.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
