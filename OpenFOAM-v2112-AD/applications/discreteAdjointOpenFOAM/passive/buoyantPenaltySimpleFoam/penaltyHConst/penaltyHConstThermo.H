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
    Foam::penaltyHConstThermo

Group
    grpSpecieThermo

Description
    Constant properties thermodynamics package
    templated into the EquationOfState.

SourceFiles
    penaltyHConstThermoI.H
    penaltyHConstThermo.C

\*---------------------------------------------------------------------------*/

#ifndef penaltyHConstThermo_H
#define penaltyHConstThermo_H

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// Forward declaration of friend functions and operators

template<class EquationOfState> class penaltyHConstThermo;

template<class EquationOfState>
inline penaltyHConstThermo<EquationOfState> operator+
(
    const penaltyHConstThermo<EquationOfState>&,
    const penaltyHConstThermo<EquationOfState>&
);

template<class EquationOfState>
inline penaltyHConstThermo<EquationOfState> operator*
(
    const scalar,
    const penaltyHConstThermo<EquationOfState>&
);

template<class EquationOfState>
inline penaltyHConstThermo<EquationOfState> operator==
(
    const penaltyHConstThermo<EquationOfState>&,
    const penaltyHConstThermo<EquationOfState>&
);

template<class EquationOfState>
Ostream& operator<<
(
    Ostream&,
    const penaltyHConstThermo<EquationOfState>&
);


/*---------------------------------------------------------------------------*\
                           Class penaltyHConstThermo Declaration
\*---------------------------------------------------------------------------*/

template<class EquationOfState>
class penaltyHConstThermo
:
    public EquationOfState
{
    // Private data

        scalar Cp_;
        scalar Hf_;

        //- Reference temperature
        scalar Tref_;

        //- Reference sensible enthalpy
        scalar Hsref_;


    // Private Member Functions

        //- Construct from components
        inline penaltyHConstThermo
        (
            const EquationOfState& st,
            const scalar cp,
            const scalar hf,
            const scalar Tref,
            const scalar Href
        );


public:

    // Constructors

        //- Construct from dictionary
        penaltyHConstThermo(const dictionary& dict);

        //- Construct as named copy
        inline penaltyHConstThermo(const word&, const penaltyHConstThermo&);

        //- Construct and return a clone
        inline autoPtr<penaltyHConstThermo> clone() const;

        //- Selector from dictionary
        inline static autoPtr<penaltyHConstThermo> New(const dictionary& dict);


    // Member Functions

        //- Return the instantiated type name
        static word typeName()
        {
            return "penaltyHConst<" + EquationOfState::typeName() + '>';
        }

        //- Limit temperature to be within the range
        inline scalar limit(const scalar T) const;


        // Fundamental properties

            //- Heat capacity at constant pressure [J/(kg K)]
            inline scalar Cp(const scalar p, const scalar T) const;

            //- Absolute Enthalpy [J/kg]
            inline scalar Ha(const scalar p, const scalar T) const;

            //- Sensible enthalpy [J/kg]
            inline scalar Hs(const scalar p, const scalar T) const;

            //- Chemical enthalpy [J/kg]
            inline scalar Hc() const;

            //- Entropy [J/(kg K)]
            inline scalar S(const scalar p, const scalar T) const;

            //- Gibbs free energy of the mixture in the standard state [J/kg]
            inline scalar Gstd(const scalar T) const;

            #include "HtoEthermo.H"


        // Derivative term used for Jacobian


            //- Temperature derivative of heat capacity at constant pressure
            inline scalar dCpdT(const scalar p, const scalar T) const;



        // I-O

            //- Write to Ostream
            void write(Ostream& os) const;


    // Member operators

        inline void operator+=(const penaltyHConstThermo&);


    // Friend operators

        friend penaltyHConstThermo operator+ <EquationOfState>
        (
            const penaltyHConstThermo&,
            const penaltyHConstThermo&
        );

        friend penaltyHConstThermo operator* <EquationOfState>
        (
            const scalar,
            const penaltyHConstThermo&
        );

        friend penaltyHConstThermo operator== <EquationOfState>
        (
            const penaltyHConstThermo&,
            const penaltyHConstThermo&
        );


    // IOstream Operators

        friend Ostream& operator<< <EquationOfState>
        (
            Ostream&,
            const penaltyHConstThermo&
        );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "penaltyHConstThermoI.H"

#ifdef NoRepository
    #include "penaltyHConstThermo.C"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
