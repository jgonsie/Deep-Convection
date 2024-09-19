/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019-2021 OpenCFD Ltd.
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
    Write primitive and binary block from OPstream

\*---------------------------------------------------------------------------*/

#include "UOPstream.H"
#include "PstreamGlobals.H"
#include "profilingPstream.H"

#include <mpi.h>

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

#include <cxxabi.h>
#if defined(DAOF_AD_MODE_A1S)
// AMPI version
bool Foam::UOPstream::write
(
    const commsTypes commsType,
    const int toProcNo,
    const char* buf,
    const std::streamsize bufSize,
    const std::string& caller,
    const std::type_info& sendType,
    const int tag,
    const label communicator
)
{
    if (debug)
    {
        Pout<< "UOPstream::write : starting write to:" << toProcNo
            << " tag:" << tag
            << " comm:" << communicator << " size:" << label(bufSize)
            << " commsType:" << UPstream::commsTypeNames[commsType]
            << Foam::endl;

        int status;
        char* realname = abi::__cxa_demangle(sendType.name(), 0, 0, &status);
        Pout << "MPI::UOPstream::write : " << caller << " type: " << realname << " size: " << std::to_string(bufSize) << Foam::endl;
        Pout << "MPI::write | " << realname << " | " << std::to_string(bufSize) << " | " << Foam::PstreamGlobals::isTypeActive(sendType) << Foam::endl;
        free(realname);
    }
    if (UPstream::warnComm != -1 && communicator != UPstream::warnComm)
    {
        Pout<< "UOPstream::write : starting write to:" << toProcNo
            << " tag:" << tag
            << " comm:" << communicator << " size:" << label(bufSize)
            << " commsType:" << UPstream::commsTypeNames[commsType]
            << " warnComm:" << UPstream::warnComm
            << Foam::endl;
        error::printStack(Pout);
    }


    PstreamGlobals::checkCommunicator(communicator, toProcNo);


    bool transferFailed = true;
    bool typeIsActive = AD::isTapeActive() && Foam::PstreamGlobals::isTypeActive(sendType);

    if (commsType == commsTypes::blocking)
    {
        if(typeIsActive){
            transferFailed = AMPI_Bsend
            (
                reinterpret_cast<scalar*>(const_cast<char*>(buf)),
                bufSize/sizeof(scalar),
                AMPI_DOUBLE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator] //MPI_COMM_WORLD
            );
        }else{
            transferFailed = AMPI_Bsend
            (
                const_cast<char*>(buf),
                bufSize,
                MPI_BYTE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator] //MPI_COMM_WORLD
            );
        }

        if (debug)
        {
            Pout<< "UOPstream::write : finished write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << Foam::endl;
        }
    }
    else if (commsType == commsTypes::scheduled)
    {
        if(typeIsActive){
            transferFailed = AMPI_Send
            (
                reinterpret_cast<scalar*>(const_cast<char*>(buf)),
                bufSize/sizeof(scalar),
                AMPI_DOUBLE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator] //MPI_COMM_WORLD
            );
        }else{
            transferFailed = AMPI_Send
            (
                const_cast<char*>(buf),
                bufSize,
                MPI_BYTE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator] //MPI_COMM_WORLD
            );
        }

        if (debug)
        {
            Pout<< "UOPstream::write : finished write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << Foam::endl;
        }
    }
    else if (commsType == commsTypes::nonBlocking)
    {
        AMPI_Request request;

        if(typeIsActive){
            transferFailed = AMPI_Isend
            (
                reinterpret_cast<scalar*>(const_cast<char*>(buf)),
                bufSize/sizeof(scalar),
                AMPI_DOUBLE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator], //MPI_COMM_WORLD
                &request
            );
        }else{
            transferFailed = AMPI_Isend
            (
                const_cast<char*>(buf),
                bufSize,
                MPI_BYTE,
                toProcNo,   //procID(toProcNo),
                tag,
                PstreamGlobals::MPICommunicators_[communicator], //MPI_COMM_WORLD
                &request
            );
        }

        if (debug)
        {
            Pout<< "UOPstream::write : started write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << " request:" << PstreamGlobals::outstandingRequests_.size()
                << Foam::endl;
        }

        PstreamGlobals::outstandingRequests_.append(request);
    }
    else
    {
        FatalErrorIn
        (
            "UOPstream::write"
            "(const int fromProcNo, char* buf, std::streamsize bufSize"
            ", const int)"
        )   << "Unsupported communications type "
            << UPstream::commsTypeNames[commsType]
            << Foam::abort(FatalError);
    }

    return !transferFailed;
}
#else
// passive version
bool Foam::UOPstream::write
(
    const commsTypes commsType,
    const int toProcNo,
    const char* buf,
    const std::streamsize bufSize,
    const std::string& caller,
    const std::type_info& sendType,
    const int tag,
    const label communicator
)
{
    if (debug)
    {
        Pout<< "UOPstream::write:\t starting write to:\t" << toProcNo
            << " tag:\t" << tag
            << " comm:\t" << communicator << " send size:\t" << label(bufSize)
            << " commsType:\t" << UPstream::commsTypeNames[commsType]
            << " caller: " << caller
            << Foam::endl;
    }
    if (UPstream::warnComm != -1 && communicator != UPstream::warnComm)
    {
        Pout<< "UOPstream::write:\t starting write to:\t" << toProcNo
            << " tag:\t" << tag
            << " comm:\t" << communicator << " size:\t" << label(bufSize)
            << " commsType:\t" << UPstream::commsTypeNames[commsType]
            << " warnComm:\t" << UPstream::warnComm
            << Foam::endl;
        error::printStack(Pout);
    }


    PstreamGlobals::checkCommunicator(communicator, toProcNo);


    bool transferFailed = true;

    profilingPstream::beginTiming();

    if (commsType == commsTypes::blocking)
    {
        transferFailed = MPI_Bsend
        (
            const_cast<char*>(buf),
            bufSize,
            MPI_BYTE,
            toProcNo,
            tag,
            PstreamGlobals::MPICommunicators_[communicator]
        );

        // Assume these are from scatters ...
        profilingPstream::addScatterTime();

        if (debug)
        {
            Pout<< "UOPstream::write : finished write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << Foam::endl;
        }
    }
    else if (commsType == commsTypes::scheduled)
    {
        transferFailed = MPI_Send
        (
            const_cast<char*>(buf),
            bufSize,
            MPI_BYTE,
            toProcNo,
            tag,
            PstreamGlobals::MPICommunicators_[communicator]
        );

        // Assume these are from scatters ...
        profilingPstream::addScatterTime();

        if (debug)
        {
            Pout<< "UOPstream::write : finished write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << Foam::endl;
        }
    }
    else if (commsType == commsTypes::nonBlocking)
    {
        MPI_Request request;

        transferFailed = MPI_Isend
        (
            const_cast<char*>(buf),
            bufSize,
            MPI_BYTE,
            toProcNo,
            tag,
            PstreamGlobals::MPICommunicators_[communicator],
            &request
        );

        profilingPstream::addWaitTime();

        if (debug)
        {
            Pout<< "UOPstream::write : started write to:" << toProcNo
                << " tag:" << tag << " size:" << label(bufSize)
                << " commsType:" << UPstream::commsTypeNames[commsType]
                << " request:" << PstreamGlobals::outstandingRequests_.size()
                << Foam::endl;
        }

        PstreamGlobals::outstandingRequests_.append(request);
    }
    else
    {
        FatalErrorInFunction
            << "Unsupported communications type "
            << UPstream::commsTypeNames[commsType]
            << Foam::abort(FatalError);
    }

    return !transferFailed;
}
#endif


// ************************************************************************* //
