/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011 OpenFOAM Foundation
    Copyright (C) 2018-2021 OpenCFD Ltd.
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

#ifndef doubleFloat_H
#define doubleFloat_H

#include "label.H"
#include "ADdefines.H"
#include "products.H"

#include <cmath>
//#include "ADdefines.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

//- Compare two values for equality
template<class T>
inline bool equal(const T& s1, const T& s2)
{
    return s1 == s2;
}

#define MAXMINPOW(retType, type1, type2)                                               \
                                                                                       \
inline retType max(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 > s2)? s1 : s2;                                                         \
}                                                                                      \
                                                                                       \
inline retType min(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 < s2)? s1: s2;                                                          \
}                                                                                      \
                                                                                       \
inline retType pow(const type1 s, const type2 e)                                       \
{                                                                                      \
    return ::pow(s,e);                                                                 \
}

#define MAXMINPOWAD(retType, type1, type2)                                             \
                                                                                       \
inline retType max(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 > s2)? s1 : s2;                                                         \
}                                                                                      \
                                                                                       \
inline retType min(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 < s2)? s1: s2;                                                          \
}                                                                                      \
                                                                                       \
inline retType pow(const type1 s, const type2 e)                                       \
{                                                                                      \
    return AD::operations::pow(s,e);                                                    \
}

#define MAXMINAD(retType, type1, type2)                                                \
                                                                                       \
inline retType max(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 > s2)? s1 : s2;                                                         \
}                                                                                      \
                                                                                       \
inline retType min(const type1 s1, const type2 s2)                                     \
{                                                                                      \
    return (s1 < s2)? s1: s2;                                                          \
}                                                                                      \


#ifdef DAOF_MODE_ACTIVE
// Towara: explicit for dco_type's
//#ifndef DAOF_AD_TOOL_CODI
  using AD::operations::pow;
//#endif
MAXMINAD(ADtype, ADtype, ADtype)
MAXMINAD(ADtype, ADtype, double)
MAXMINAD(ADtype, double, ADtype)
MAXMINAD(ADtype, ADtype, float)
MAXMINAD(ADtype, float, ADtype)
MAXMINAD(ADtype, ADtype, int)
MAXMINAD(ADtype, int, ADtype)
MAXMINAD(ADtype, ADtype, long)
MAXMINAD(ADtype, long, ADtype)
#endif

// Towara: for passive types
MAXMINPOW(double, double, double)
MAXMINPOW(double, double, float)
MAXMINPOW(double, float, double)
MAXMINPOW(double, double, int)
MAXMINPOW(double, int, double)
MAXMINPOW(double, double, long)
MAXMINPOW(double, long, double)
MAXMINPOW(float, float, float)
MAXMINPOW(float, float, int)
MAXMINPOW(float, int, float)
MAXMINPOW(float, float, long)
MAXMINPOW(float, long, float)
#if defined(__APPLE__) && WM_LABEL_SIZE == 64
MAXMINPOW(double, double, int64_t)
MAXMINPOW(double, int64_t, double)
MAXMINPOW(float, float, int64_t)
MAXMINPOW(float, int64_t, float)
#endif

#undef MAXMINPOW
#undef MAXMINPOWAD
#undef MAXMINAD


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
