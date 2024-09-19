/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
    Copyright (C) 2017-2021 OpenCFD Ltd.
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
    Foam::scalar

Description
    A floating-point number identical to float or double depending on
    whether WM_SP, WM_SPDP or WM_DP is defined.

SourceFiles
    scalar.C

\*---------------------------------------------------------------------------*/

#ifndef scalar_H
#define scalar_H

#include "floatScalar.H"
#include "doubleScalar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Typedefs (floatScalar, doubleScalar, scalar, solveScalar) in scalarFwd.H

#if defined(WM_SP) || defined(WM_SPDP)

// With scalar == (float), solveScalar == (float | double)

namespace Foam
{
    typedef floatScalar scalar;
    #if defined(WM_SPDP)
    typedef doubleScalar solveScalar;
    #else
    typedef floatScalar solveScalar;
    #endif
    typedef floatScalarPassive scalarPassive;

    static const scalar GREAT = floatScalarGREAT;
    static const scalar VGREAT = floatScalarVGREAT;
    static const scalar ROOTVGREAT = floatScalarROOTVGREAT;
    static const scalar SMALL = floatScalarSMALL;
    static const scalar ROOTSMALL = floatScalarROOTSMALL;
    static const scalar VSMALL = floatScalarVSMALL;
    static const scalar ROOTVSMALL = floatScalarROOTVSMALL;

    #ifdef COMPAT_OPENFOAM_ORG
    //! \cond compat_openfoam_org
    //  Accommodate name changes from 2018-01
    constexpr scalar great = floatScalarGREAT;
    constexpr scalar rootGreat = floatScalarROOTGREAT;
    constexpr scalar vGreat = floatScalarVGREAT;
    constexpr scalar rootVGreat = floatScalarROOTVGREAT;
    constexpr scalar small = floatScalarSMALL;
    constexpr scalar rootSmall = floatScalarROOTSMALL;
    constexpr scalar vSmall = floatScalarVSMALL;
    constexpr scalar rootVSmall = floatScalarROOTVSMALL;
    //! \endcond
    #endif


    //- Read scalar from c-string and return value
    inline scalar readScalar(const char* buf)
    {
        return readFloat(buf);
    }

    //- Read scalar from c-string into argument. Return true on success.
    inline bool readScalar(const char* buf, scalar& val)
    {
        return readFloat(buf, val);
    }

    //- Read scalar from string and return value
    inline scalar readScalar(const std::string& str)
    {
        return readFloat(str);
    }

    //- Read scalar from string into argument. Return true on success.
    inline bool readScalar(const std::string& str, scalar& val)
    {
        return readFloat(str, val);
    }

    //- Read scalar from stream.
    scalar readScalar(Istream& is);

    //- Read raw scalar from binary stream.
    //  \note No internal check for binary vs ascii,
    //        the caller knows what they are doing
    scalar readRawScalar(Istream& is);

    //- Read raw scalar(s) from binary stream.
    //  \note No internal check for binary vs ascii,
    //        the caller knows what they are doing
    void readRawScalar(Istream& is, scalar* data, size_t nElem = 1);
}

#elif defined(WM_DP)

// With scalar == (double), solveScalar == (double)

namespace Foam
{
    static const scalar GREAT = doubleScalarGREAT;
    static const scalar VGREAT = doubleScalarVGREAT;
    static const scalar ROOTVGREAT = doubleScalarROOTVGREAT;
    static const scalar SMALL = doubleScalarSMALL;
    static const scalar ROOTSMALL = doubleScalarROOTSMALL;
    static const scalar VSMALL = doubleScalarVSMALL;
    static const scalar ROOTVSMALL = doubleScalarROOTVSMALL;

    #ifdef COMPAT_OPENFOAM_ORG
    //! \cond compat_openfoam_org
    //  Accommodate name changes from 2018-01
    constexpr scalar great = doubleScalarGREAT;
    constexpr scalar rootGreat = doubleScalarROOTGREAT;
    constexpr scalar vGreat = doubleScalarVGREAT;
    constexpr scalar rootVGreat = doubleScalarROOTVGREAT;
    constexpr scalar small = doubleScalarSMALL;
    constexpr scalar rootSmall = doubleScalarROOTSMALL;
    constexpr scalar vSmall = doubleScalarVSMALL;
    constexpr scalar rootVSmall = doubleScalarROOTVSMALL;
    //! \endcond
    #endif


    //- Read scalar from c-string and return value
    inline scalar readScalar(const char* buf)
    {
        return readDouble(buf);
    }

    //- Read scalar from c-string into argument. Return true on success.
    inline bool readScalar(const char* buf, scalar& val)
    {
        return readDouble(buf, val);
    }

    //- Read scalar from string and return value
    inline scalar readScalar(const std::string& str)
    {
        return readDouble(str);
    }

    //- Read scalar from string into argument. Return true on success.
    inline bool readScalar(const std::string& str, scalar& val)
    {
        return readDouble(str, val);
    }


    //- Read scalar from stream.
    scalar readScalar(Istream& is);

    //- Read raw scalar from binary stream.
    //  \note No internal check for binary vs ascii,
    //        the caller knows what they are doing
    scalar readRawScalar(Istream& is);

    //- Read raw scalar(s) from binary stream.
    //  \note No internal check for binary vs ascii,
    //        the caller knows what they are doing
    void readRawScalar(Istream& is, scalar* data, size_t nElem = 1);
}

#else
//    #error "PRECISION must be set to WM_SP, WM_SPDP or WM_DP"
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Type conversions (narrowing)

namespace Foam
{

//- Type narrowing from double to float
//  Overflow: silently fix, or raise error?
inline float narrowFloat(const double val)
{
    // Single statement - future constexpr?
    return
    (
        (val <= -floatScalarVGREAT) ? -floatScalarVGREAT
      : (val >=  floatScalarVGREAT) ?  floatScalarVGREAT
      : (val > -floatScalarVSMALL && val < floatScalarVSMALL) // underflow
      ? 0
      : static_cast<float>(val)
    );
}

} // End namespace Foam


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    //- Type to use for extended precision
    template<>
    class typeOfSolve<scalar>
    {
    public:

        typedef solveScalar type;
    };
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
