/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2012-2015 OpenFOAM Foundation
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

\*---------------------------------------------------------------------------*/
#include <mpi.h>
#include "allReduce.H"
#include "profilingPstream.H"

// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //
#if defined(DAOF_AD_MODE_A1S)
// AMPI version
template<class Type, class BinaryOp>
void Foam::allReduce
(
    Type& Value,
    int MPICount,
    MPI_Datatype MPIType,
    MPI_Op MPIOp,
    const BinaryOp& bop,
    const int tag,
    const label communicator
)
{
    if (!UPstream::parRun())
    {
        return;
    }
    if(UPstream::debug)
    {
        Pout << "Foam::allReduce with AMPI" << endl;
    }

    profilingPstream::beginTiming();

    if (UPstream::nProcs(communicator) <= UPstream::nProcsSimpleSum)
    {
        if (UPstream::master(communicator))
        {
            for (const int proci : UPstream::subProcs(communicator))
            {
                Type value;

                if
                (
                    AMPI_Recv
                    (
                        &value,
                        MPICount,
                        MPIType,
                        proci,
                        tag,
                        PstreamGlobals::MPICommunicators_[communicator],
                        MPI_STATUS_IGNORE
                    )
                )
                {
                    FatalErrorIn
                    (
                        "void Foam::allReduce\n"
                        "(\n"
                        "    Type&,\n"
                        "    int,\n"
                        "    MPI_Datatype,\n"
                        "    MPI_Op,\n"
                        "    const BinaryOp&,\n"
                        "    const int\n"
                        ")\n"
                    )   << "MPI_Recv failed"
                        << Foam::abort(FatalError);
                }

                Value = bop(Value, value);
            }
        }
        else
        {
            if
            (
                AMPI_Send
                (
                    &Value,
                    MPICount,
                    MPIType,
                    UPstream::masterNo(),//UPstream::procID(masterNo()),
                    tag,
                    PstreamGlobals::MPICommunicators_[communicator]
                )
            )
            {
                FatalErrorIn
                (
                    "void Foam::allReduce\n"
                    "(\n"
                    "    Type&,\n"
                    "    int,\n"
                    "    MPI_Datatype,\n"
                    "    MPI_Op,\n"
                    "    const BinaryOp&,\n"
                    "    const int\n"
                    ")\n"
                )   << "MPI_Send failed"
                    << Foam::abort(FatalError);
            }
        }


        if (UPstream::master(communicator))
        {
            for (const int proci : UPstream::subProcs(communicator))
            {
                if
                (
                    AMPI_Send
                    (
                        &Value,
                        MPICount,
                        MPIType,
                        proci,
                        tag,
                        PstreamGlobals::MPICommunicators_[communicator]
                    )
                )
                {
                    FatalErrorIn
                    (
                        "void Foam::allReduce\n"
                        "(\n"
                        "    Type&,\n"
                        "    int,\n"
                        "    MPI_Datatype,\n"
                        "    MPI_Op,\n"
                        "    const BinaryOp&,\n"
                        "    const int\n"
                        ")\n"
                    )   << "MPI_Send failed"
                        << Foam::abort(FatalError);
                }
            }
        }
        else
        {
            if
            (
                AMPI_Recv
                (
                    &Value,
                    MPICount,
                    MPIType,
                    UPstream::masterNo(),//UPstream::procID(masterNo()),
                    tag,
                    PstreamGlobals::MPICommunicators_[communicator],
                    MPI_STATUS_IGNORE
                )
            )
            {
                FatalErrorIn
                (
                    "void Foam::allReduce\n"
                    "(\n"
                    "    Type&,\n"
                    "    int,\n"
                    "    MPI_Datatype,\n"
                    "    MPI_Op,\n"
                    "    const BinaryOp&,\n"
                    "    const int\n"
                    ")\n"
                )   << "MPI_Recv failed"
                    << Foam::abort(FatalError);
            }
        }
    }
    else
    {
        Type sum;
        AMPI_Allreduce
        (
            &Value,
            &sum,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator]
        );
        Value = sum;
    }
}

template<class Type>
void Foam::iallReduce
(
    void* recvBuf,
    int MPICount,
    MPI_Datatype MPIType,
    MPI_Op MPIOp,
    const label communicator,
    label& requestID
)
{
    if (!UPstream::parRun())
    {
        return;
    }

    if (UPstream::warnComm != -1 && communicator != UPstream::warnComm)
    {
        Pout<< "** non-blocking reducing:"
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << " with comm:" << communicator
            << " warnComm:" << UPstream::warnComm << endl;
        error::printStack(Pout);
    }

    profilingPstream::beginTiming();

#if (false) && defined(MPI_VERSION) && (MPI_VERSION >= 3)
    AMPI_Request request;
    if
    (
        AMPI_Iallreduce
        (
            MPI_IN_PLACE,
            recvBuf,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator],
            &request
        )
    )
    {
        FatalErrorInFunction
            << "MPI_Iallreduce failed for "
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << Foam::abort(FatalError);
    }

    if (PstreamGlobals::freedRequests_.size())
    {
        requestID = PstreamGlobals::freedRequests_.remove();
        PstreamGlobals::outstandingRequests_[requestID] = request;
    }
    else
    {
        requestID = PstreamGlobals::outstandingRequests_.size();
        PstreamGlobals::outstandingRequests_.append(request);
    }

    if (UPstream::debug)
    {
        Pout<< "UPstream::allocateRequest for non-blocking reduce"
            << " : request:" << requestID << endl;
    }
#else
    // Non-blocking not yet implemented in mpi
    if
    (
        AMPI_Allreduce
        (
            MPI_IN_PLACE,
            recvBuf,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator]
        )
    )
    {
        FatalErrorInFunction
            << "MPI_Allreduce failed for "
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << Foam::abort(FatalError);
    }
    requestID = -1;
#endif

    profilingPstream::addReduceTime();
}
#else
template<class Type, class BinaryOp>
void Foam::allReduce
(
    Type& Value,
    int MPICount,
    MPI_Datatype MPIType,
    MPI_Op MPIOp,
    const BinaryOp& bop,
    const int tag,
    const label communicator
)
{
    if (!UPstream::parRun())
    {
        return;
    }

    // Towara: EXPERIMENTAL enable mpi for t1s/t1v mode
    bool forceSimpleSum = false;
    #if defined(DAOF_AD_MODE_T1S) || defined(DAOF_AD_MODE_T1V)
        if(MPIType == MPI_DOUBLE)
            MPICount *= sizeof(Foam::scalar)/sizeof(double);
        forceSimpleSum = (MPIOp != MPI_SUM); // enforce simpleSum for everything but sumOp
    #endif

    if (UPstream::nProcs(communicator) <= UPstream::nProcsSimpleSum || forceSimpleSum)
    {
        //Info << "allReduce simpleSum "    << ( MPIOp == MPI_SUM ) << " " << typeid(bop).name() << " count:" << MPICount << endl;
        if (UPstream::master(communicator))
        {
            for (const int proci : UPstream::subProcs(communicator))
            {
                Type value;

                if
                (
                    MPI_Recv
                    (
                        &value,
                        MPICount,
                        MPIType,
                        proci,
                        tag,
                        PstreamGlobals::MPICommunicators_[communicator],
                        MPI_STATUS_IGNORE
                    )
                )
                {
                    FatalErrorInFunction
                        << "MPI_Recv failed"
                        << Foam::abort(FatalError);
                }

                Value = bop(Value, value);
            }
        }
        else
        {
            if
            (
                MPI_Send
                (
                    &Value,
                    MPICount,
                    MPIType,
                    UPstream::masterNo(),
                    tag,
                    PstreamGlobals::MPICommunicators_[communicator]
                )
            )
            {
                FatalErrorInFunction
                    << "MPI_Send failed"
                    << Foam::abort(FatalError);
            }
        }


        if (UPstream::master(communicator))
        {
            for (const int proci : UPstream::subProcs(communicator))
            {
                if
                (
                    MPI_Send
                    (
                        &Value,
                        MPICount,
                        MPIType,
                        proci,
                        tag,
                        PstreamGlobals::MPICommunicators_[communicator]
                    )
                )
                {
                    FatalErrorInFunction
                        << "MPI_Send failed"
                        << Foam::abort(FatalError);
                }
            }
        }
        else
        {
            if
            (
                MPI_Recv
                (
                    &Value,
                    MPICount,
                    MPIType,
                    UPstream::masterNo(),
                    tag,
                    PstreamGlobals::MPICommunicators_[communicator],
                    MPI_STATUS_IGNORE
                )
            )
            {
                FatalErrorInFunction
                    << "MPI_Recv failed"
                    << Foam::abort(FatalError);
            }
        }
    }
    else
    {
        Type sum;
        MPI_Allreduce
        (
            &Value,
            &sum,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator]
        );
        Value = sum;
    }

    profilingPstream::addReduceTime();
}

template<class Type>
void Foam::iallReduce
(
    void* recvBuf,
    int MPICount,
    MPI_Datatype MPIType,
    MPI_Op MPIOp,
    const label communicator,
    label& requestID
)
{
    if (!UPstream::parRun())
    {
        return;
    }

    if (UPstream::warnComm != -1 && communicator != UPstream::warnComm)
    {
        Pout<< "** non-blocking reducing:"
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << " with comm:" << communicator
            << " warnComm:" << UPstream::warnComm << endl;
        error::printStack(Pout);
    }

    profilingPstream::beginTiming();

#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
    MPI_Request request;
    if
    (
        MPI_Iallreduce
        (
            MPI_IN_PLACE,
            recvBuf,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator],
            &request
        )
    )
    {
        FatalErrorInFunction
            << "MPI_Iallreduce failed for "
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << Foam::abort(FatalError);
    }

    if (PstreamGlobals::freedRequests_.size())
    {
        requestID = PstreamGlobals::freedRequests_.remove();
        PstreamGlobals::outstandingRequests_[requestID] = request;
    }
    else
    {
        requestID = PstreamGlobals::outstandingRequests_.size();
        PstreamGlobals::outstandingRequests_.append(request);
    }

    if (UPstream::debug)
    {
        Pout<< "UPstream::allocateRequest for non-blocking reduce"
            << " : request:" << requestID << endl;
    }
#else
    // Non-blocking not yet implemented in mpi
    if
    (
        MPI_Allreduce
        (
            MPI_IN_PLACE,
            recvBuf,
            MPICount,
            MPIType,
            MPIOp,
            PstreamGlobals::MPICommunicators_[communicator]
        )
    )
    {
        FatalErrorInFunction
            << "MPI_Allreduce failed for "
            << UList<Type>(static_cast<Type*>(recvBuf), MPICount)
            << Foam::abort(FatalError);
    }
    requestID = -1;
#endif

    profilingPstream::addReduceTime();
}
#endif


// ************************************************************************* //
