/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2016 OpenFOAM Foundation
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

Class
    Foam::blockVertices::derivPointVertex

Description

SourceFiles
    derivPointVertex.C

\*---------------------------------------------------------------------------*/

#ifndef blockVertices_derivPointVertex_H
#define blockVertices_derivPointVertex_H

#include "pointVertex.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace blockVertices
{

/*---------------------------------------------------------------------------*\
                           Class derivPointVertex Declaration
\*---------------------------------------------------------------------------*/

class derivPointVertex
:
    public pointVertex
{
public:

    //- Runtime type information
    TypeName("derivPoint");


    // Constructors
    //- Construct from Istream setting pointsList
    derivPointVertex
    (
        const dictionary&,
        const label index,
        const searchableSurfaces& geometry,
        Istream&
    );


    //- Destructor
    virtual ~derivPointVertex()
    {}
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace blockVertices
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
