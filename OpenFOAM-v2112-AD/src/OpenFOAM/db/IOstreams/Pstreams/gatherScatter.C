/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
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

Description
    Gather data from all processors onto single processor according to some
    communication schedule (usually linear-to-master or tree-to-master).
    The gathered data will be a single value constructed from the values
    on individual processors using a user-specified operator.

\*---------------------------------------------------------------------------*/

#include "UOPstream.H"
#include "OPstream.H"
#include "UIPstream.H"
#include "IPstream.H"
#include "contiguous.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

template<class T,class B>
T Pstream::myBop(T& v1,T v2, const B& bop){
    return bop(v1, v2);
}
template<class B>
Foam::label Pstream::myBop(Foam::label& v1,Foam::label v2, const B& bop){
    return Foam::label(AD::passiveValue(bop(v1, v2)));
}

template<class T, class BinaryOp>
void Pstream::gather
(
    const List<UPstream::commsStruct>& comms,
    T& Value,
    const BinaryOp& bop,
    const int tag,
    const label comm
)
{
    if (UPstream::parRun() && UPstream::nProcs(comm) > 1)
    {
        // Get my communication order
        const commsStruct& myComm = comms[UPstream::myProcNo(comm)];

        // Receive from my downstairs neighbours
        forAll(myComm.below(), belowI)
        {
            T value;

            if (is_contiguous<T>::value)
            {
                UIPstream::read
                (
                    UPstream::commsTypes::scheduled,
                    myComm.below()[belowI],
                    reinterpret_cast<char*>(&value),
                    sizeof(T),
                    pretty_print(),
                    typeid(&value),
                    tag,
                    comm
                );
            }
            else
            {
                IPstream fromBelow
                (
                    UPstream::commsTypes::scheduled,
                    myComm.below()[belowI],
                    0,
                    tag,
                    comm
                );
                fromBelow >> value;
            }
            // Towara: fix for label reduction
            Value = myBop(Value, value,bop);
        }

        // Send up Value
        if (myComm.above() != -1)
        {
            if (is_contiguous<T>::value)
            {
                UOPstream::write
                (
                    UPstream::commsTypes::scheduled,
                    myComm.above(),
                    reinterpret_cast<const char*>(&Value),
                    sizeof(T),
                    pretty_print(),
                    typeid(&Value),
                    tag,
                    comm
                );
            }
            else
            {
                OPstream toAbove
                (
                    UPstream::commsTypes::scheduled,
                    myComm.above(),
                    0,
                    tag,
                    comm
                );
                toAbove << Value;
            }
        }
    }
}


template<class T, class BinaryOp>
void Pstream::gather
(
    T& Value,
    const BinaryOp& bop,
    const int tag,
    const label comm
)
{
    if (UPstream::nProcs(comm) < UPstream::nProcsSimpleSum)
    {
        gather(UPstream::linearCommunication(comm), Value, bop, tag, comm);
    }
    else
    {
        gather(UPstream::treeCommunication(comm), Value, bop, tag, comm);
    }
}


template<class T>
void Pstream::scatter
(
    const List<UPstream::commsStruct>& comms,
    T& Value,
    const int tag,
    const label comm
)
{
    if (UPstream::parRun() && UPstream::nProcs(comm) > 1)
    {
        // Get my communication order
        const commsStruct& myComm = comms[UPstream::myProcNo(comm)];

        // Receive from up
        if (myComm.above() != -1)
        {
            if (is_contiguous<T>::value)
            {
                UIPstream::read
                (
                    UPstream::commsTypes::scheduled,
                    myComm.above(),
                    reinterpret_cast<char*>(&Value),
                    sizeof(T),
                    pretty_print(),
                    typeid(&Value),
                    tag,
                    comm
                );
            }
            else
            {
                IPstream fromAbove
                (
                    UPstream::commsTypes::scheduled,
                    myComm.above(),
                    0,
                    tag,
                    comm
                );
                fromAbove >> Value;
            }
        }

        // Send to my downstairs neighbours. Note reverse order (compared to
        // receiving). This is to make sure to send to the critical path
        // (only when using a tree schedule!) first.
        forAllReverse(myComm.below(), belowI)
        {
            if (is_contiguous<T>::value)
            {
                UOPstream::write
                (
                    UPstream::commsTypes::scheduled,
                    myComm.below()[belowI],
                    reinterpret_cast<const char*>(&Value),
                    sizeof(T),
                    pretty_print(),
                    typeid(&Value),
                    tag,
                    comm
                );
            }
            else
            {
                OPstream toBelow
                (
                    UPstream::commsTypes::scheduled,
                    myComm.below()[belowI],
                    0,
                    tag,
                    comm
                );
                toBelow << Value;
            }
        }
    }
}


template<class T>
void Pstream::scatter(T& Value, const int tag, const label comm)
{
    if (UPstream::nProcs(comm) < UPstream::nProcsSimpleSum)
    {
        scatter(UPstream::linearCommunication(comm), Value, tag, comm);
    }
    else
    {
        scatter(UPstream::treeCommunication(comm), Value, tag, comm);
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //
