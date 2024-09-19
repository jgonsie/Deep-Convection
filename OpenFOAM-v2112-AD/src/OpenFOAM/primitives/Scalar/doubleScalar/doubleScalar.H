/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2015 OpenFOAM Foundation
    Copyright (C) 2017-2020 OpenCFD Ltd.
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

Typedef
    Foam::doubleScalar

Description
    Floating-point double precision scalar type.

SourceFiles
    doubleScalar.C

\*---------------------------------------------------------------------------*/

#ifndef doubleScalar_H
#define doubleScalar_H

#include "scalarFwd.H"
#include "doubleFloat.H"
#include "direction.H"
#include "word.H"

// change for discreteAdjointOpenFOAM by Towara@STCE: include dco headers
// Towara: also include helper functions, thus making them globally available
#include "ADdefines.H"
#include "stce_helpers.C"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// change for discreteAdjointOpenFOAM by Towara@STCE: implement transFuncs in AD::operations namespace
#ifdef DAOF_AD_TOOL_CODI
namespace codi{
        template<class T>
        inline Foam::doubleScalar log1p(const T s){
            return log(1.0+s);
        }
        template<class T>
        inline Foam::doubleScalar log10(const T s){
            return log(s)/std::log(10);
        }
        template<class T>
        inline Foam::doubleScalar asinh(const T s){
            return log(s+sqrt(pow(s,2)+1));
        }
        template<class T>
        inline Foam::doubleScalar acosh(const T s){
            return log(s+sqrt(pow(s,2)+1));
        }
        template<class T>
        inline Foam::doubleScalar cbrt(const T s){
            return pow(s,1.0/3.0);
        }
        template<class T>
        inline T lgamma(const T s){
            return static_cast<T>(std::lgamma(AD::passiveValue(s)));
        }
        template<class T>
        inline T tgamma(const T s){
            return static_cast<T>(std::tgamma(AD::passiveValue(s)));
        }
        template<class T>
        inline T j0(const T s){
            return static_cast<T>(::j0(AD::passiveValue(s)));
        }
        template<class T>
        inline T j1(const T s){
            return static_cast<T>(::j1(AD::passiveValue(s)));
        }
        template<class T>
        inline T y0(const T s){
            return static_cast<T>(::y0(AD::passiveValue(s)));
        }
        template<class T>
        inline T y1(const T s){
            return static_cast<T>(::y1(AD::passiveValue(s)));
        }
        template<class T>
        inline T fmod(const T x, const T y){
            // Towara: hacky way to implement active fmod
            return static_cast<T>(x - Foam::label(AD::passiveValue(x/y))*y);
        }
}
#endif


namespace Foam
{

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Largest and smallest scalar values allowed in certain parts of the code.
// See std::numeric_limits max(), min(), epsilon()
constexpr double  doubleScalarGREAT = 1.0e+15;
constexpr double  doubleScalarVGREAT = 1.0e+300;
constexpr double  doubleScalarROOTVGREAT = 1.0e+150;
constexpr double  doubleScalarSMALL = 1.0e-15;
constexpr double  doubleScalarROOTSMALL = 3.0e-8;
constexpr double  doubleScalarVSMALL = 1.0e-300;
constexpr double  doubleScalarROOTVSMALL = 1.0e-150;


#define Scalar doubleScalar
#define ScalarPassive doubleScalarPassive
#define ScalarVGREAT doubleScalarVGREAT
#define ScalarVSMALL doubleScalarVSMALL
#define ScalarROOTVGREAT doubleScalarROOTVGREAT
#define ScalarROOTVSMALL doubleScalarROOTVSMALL
#define ScalarRead readDouble


inline Scalar mag(const Scalar s)
{
    // change for discreteAdjointOpenFOAM by Towara@STCE: change namespace for fabs
    return fabs(s);
}

inline Scalar hypot(const Scalar x, const Scalar y)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
// def of hypot: http://en.cppreference.com/w/cpp/numeric/math/hypot
    return sqrt(x*x+y*y);
}

#ifdef DAOF_MODE_ACTIVE
inline Scalar atan2(const Scalar y, const Scalar x)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
    return AD::operations::atan2(y, x);
}

// Towara: implement fma (fusem multiply add)
inline Scalar fma(const Scalar x, const Scalar y, const Scalar z)
{
    return x*y+z;
}
#else
inline Scalar atan2(const Scalar y, const Scalar x)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
    return ::atan2(y, x);
}

// Towara: implement fma (fusem multiply add)
inline Scalar fma(const Scalar x, const Scalar y, const Scalar z)
{
    return std::fma(x,y,z);
}
#endif

/*inline Scalar jn(const int n, const Scalar s)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
    return ::jn(n, AD::passiveValue(s));
}

inline Scalar yn(const int n, const Scalar s)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
    return ::yn(n, AD::passiveValue(s));
}*/

// Normal (double-precision) transcendental functions
// change for discreteAdjointOpenFOAM by Towara@STCE: change definition of transFunc macro
#ifdef DAOF_MODE_ACTIVE
#define transFunc(func)              \
inline Scalar func(const Scalar s)   \
{                                    \
    return ::AD::operations::func(s); \
}
#else
#define transFunc(func)              \
inline Scalar func(const Scalar s)   \
{                                    \
    return ::func(s);                \
}
#endif


// Double-precision bessel functions
// change for discreteAdjointOpenFOAM by Towara@STCE: change definition of besselFunc macro
#ifdef DAOF_MODE_ACTIVE
#define besselFunc(func)             \
inline Scalar func(const Scalar s)   \
{                                    \
    return ::AD::operations::func(s); \
}
#else
#define besselFunc(func)             \
inline Scalar func(const Scalar s)   \
{                                    \
    return ::func(s);                \
}
#endif

#ifdef DAOF_MODE_ACTIVE
    #define bessel2Func(func)                                   \
    inline Scalar func(const int n, const Scalar s)             \
    {                                                           \
        return ::func(n, AD::passiveValue(s));                 \
    }
#else
    #define bessel2Func(func)                                   \
    inline Scalar func(const int n, const Scalar s)             \
    {                                                           \
        return ::func(n, s);                                    \
    }
#endif

// change for discreteAdjointOpenFOAM by Towara@STCE: add implementation of sqrt
#ifdef DAOF_MODE_ACTIVE
inline Scalar sqrt(const Scalar x)
{
// change for discreteAdjointOpenFOAM by Towara@STCE: change namespace
    return AD::operations::sqrt(x);
}
inline double sqrt(const double x)
{
    return std::sqrt(x);
}
#endif

// change for discreteAdjointOpenFOAM by Towara@STCE: explicit instantiation of operator<< for double
Ostream& operator<<(Ostream& os, const double s);
Istream& operator>>(Istream& is, double& val);

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#include "Scalar.H"

#undef Scalar
#undef ScalarPassive
#undef ScalarVGREAT
#undef ScalarVSMALL
#undef ScalarROOTVGREAT
#undef ScalarROOTVSMALL
#undef ScalarRead
#undef transFunc
#undef besselFunc
#undef bessel2Func


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
