/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2019-2020 OpenCFD Ltd.
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
    Foam::PetscWrappedVector

Description
    Wrap an OpenFOAM list as a PETSc vector

SourceFiles

\*---------------------------------------------------------------------------*/

#ifndef petscFoamWrappedVector_H
#define petscFoamWrappedVector_H

// OpenFOAM
#include "List.H"
#include "scalar.H"
#include "vector.H"

// PETSc
#include "petscvec.h"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                     Class PetscWrappedVector Declaration
\*---------------------------------------------------------------------------*/

class PetscWrappedVector
{
    // Private Data

        //- The PETSc vector
        Vec vec_;

        // passive copy of data
        scalarPassive *data_;
        // pointer to OpenFOAM data
        scalar *OFdata_;
        label vecSize_;

        //- Call VecResetArray on destruction?
        bool reset_;


    // Private Member Functions

        //- Wrap an OpenFOAM list of values as a PETSc vector
        template<class T>
        void createImpl
        (
            const UList<T>& list,
            Mat mat
        )
        {
            MatCreateVecs(mat, &vec_, NULL);
            // copy data to passive buffer
            vecSize_ = list.byteSize()/sizeof(scalar);
            OFdata_ = reinterpret_cast<scalar*>(const_cast<T*>(list.cdata()));
            if constexpr(!std::is_same_v<double,scalar>){
                data_ = new scalarPassive[vecSize_];
                // copy passive data
                Info << "copying data into PetSc passive format (size = " << vecSize_ << " )" << endl;
                std::transform(
                    list.cdata(),
                    list.cdata() + vecSize_,
                    data_,
                    [](const scalar& v){return AD::passiveValue(v);}
                );
            } else {
                data_ = reinterpret_cast<solveScalar*>(const_cast<T*>(list.cdata()));
            }

            VecPlaceArray
            (
                vec_,
                data_
            );
            reset_ = true;
        }

        //- Wrap an OpenFOAM list of values as a PETSc vector
        template<class T>
        void createImpl
        (
            const UList<T>& list,
            MPI_Comm comm
        )
        {
            // copy data to passive buffer
            vecSize_ = list.byteSize()/sizeof(scalar);
            OFdata_ = reinterpret_cast<scalar*>(const_cast<T*>(list.cdata()));
            if constexpr(!std::is_same_v<double,scalar>){
                data_ = new scalarPassive[vecSize_];
                // copy passive data
                Info << "copying data into PetSc passive format (MPI, size = " << vecSize_ << " )" << endl;
                std::transform(
                    list.cdata(),
                    list.cdata() + vecSize_,
                    data_,
                    [](const scalar& v){return AD::passiveValue(v);}
                );
            } else {
                data_ = reinterpret_cast<solveScalar*>(const_cast<T*>(list.cdata()));
            }
            VecCreateMPIWithArray
            (
                comm,
                pTraits<T>::nComponents,
                list.size(),
                PETSC_DECIDE,
                data_,
                &vec_
            );
            reset_ = false;
        }


    // Constructors

        //- Default construct
        PetscWrappedVector()
        :
            vec_(),
            reset_(false)
        {}


    // Generated Methods

        //- No copy construct
        PetscWrappedVector(const PetscWrappedVector&) = delete;

        //- No copy assignment
        void operator=(const PetscWrappedVector&) = delete;


public:

    // Constructors

        //- Wrap an OpenFOAM list of values as a PETSc vector
        PetscWrappedVector
        (
            const UList<solveScalar>& list,
            Mat mat
        )
        :
            PetscWrappedVector()
        {
            createImpl(list, mat);
        }

        //- Wrap an OpenFOAM list of values as a PETSc vector
        explicit PetscWrappedVector
        (
            const UList<solveScalar>& list,
            MPI_Comm comm = PETSC_COMM_WORLD
        )
        :
            PetscWrappedVector()
        {
            createImpl(list, comm);
        }


    //- Destructor
    ~PetscWrappedVector()
    {
        if (reset_)
        {
            VecResetArray(vec_);
        }
        if constexpr(!std::is_same_v<double,scalar>){
            // copy back data into active format
            Info << "copying PetSc passive format back to OpenFOAM (size = " << vecSize_ << " )" << endl;
            /*std::copy(
                data_,
                data_ + vecSize_,
                OFdata_
            );*/
            for(label i=0; i < vecSize_; i++){
                AD::value(OFdata_[i]) = data_[i];
            }
            delete []data_;
        }
        VecDestroy(&vec_);
    }


    // Member Operators

        //- Allow cast to a const PETSc vector
        operator const Vec&() const
        {
            return vec_;
        }

        //- Allow cast to a non-const PETSc vector
        operator Vec&()
        {
            return vec_;
        }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
