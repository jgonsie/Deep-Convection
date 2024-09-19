#ifndef AMPI_TAPE_INCLUDE
#define AMPI_TAPE_INCLUDE

/** \file
 * \brief ampi_tape.h provides the AMPI tape intented to be used by an
 * overloading AD tool. The AMPI tape handles and saves all the relevant MPI
 * information, while the data is kept in the trace of the AD tool. Interface
 * functions are defined that allow AMPI and the AD tool to exchange
 * information. This interface routines need to be implemented for each AD tool.
 * The MPI communications are directly mapped to the active routines defined in
 * this file by calling the forward active routines in ampi.h. In the reverse
 * section, the AD tool calls the AMPI tape through ampi_interpret_tape() when
 * it hits an AMPI operation. The AMPI tape then executes the reverse MPI
 * communication. 
 */

/** \def AMPI_DOUBLE
 * Sets the active MPI type. If all buffers of type MPI_DOUBLE should be
 * communicated actively, set this to MPI_DOUBLE
 */
#define AMPI_DOUBLE MPI_DOUBLE

/**
 * \def AMPI_CHUNK_SIZE
 * Sets the chunk size of the AMPI tape. When the tape exceeds this size, a new
 * chunk is allocated.
 */
#define AMPI_CHUNK_SIZE 500000

/**
 * @{
 * \name Internal defines for the reduction operation
 */
#define AMPI_REDUCE_ADD 1
#define AMPI_REDUCE_MUL 2
#define AMPI_REDUCE_MIN 3
#define AMPI_REDUCE_MAX 4
/**@}*/

/**
 * @{
 * \name Internal defines for the MPI communications
 */
#define AMPI_SEND 1
#define AMPI_RECV 2
#define AMPI_ISEND 3
#define AMPI_IRECV 4
#define AMPI_WAIT 5
#define AMPI_WAITALL 6
#define AMPI_AWAITALL 7
#define AMPI_BCAST 8
#define AMPI_REDUCE 9
#define AMPI_ALLREDUCE 10
#define AMPI_MPI_DUMMY 11
#define AMPI_MPI_ADUMMY 12
#define AMPI_SENDRECVREPLACE 13
#define AMPI_SCATTER 14
#define AMPI_GATHER 15
#define AMPI_SCATTERV 16
#define AMPI_GATHERV 17
#define AMPI_SEND_INIT 18
#define AMPI_RECV_INIT 19
#define AMPI_START 20
#define AMPI_STARTALL 21
#define AMPI_SENDRECV 22
#define AMPI_BRECV 23
#define AMPI_ALLGATHER 24
#define AMPI_ALLGATHERV 25

#ifdef AMPI_DEBUG
static const char* AMPI_OPCODES[] = {"AMPI_NONE","AMPI_SEND","AMPI_RECV","AMPI_ISEND","AMPI_IRECV","AMPI_WAIT","AMPI_WAITALL","AMPI_AWAITALL","AMPI_BCAST","AMPI_REDUCE","AMPI_ALLREDUCE","AMPI_MPI_DUMMY","AMPI_MPI_ADUMMY","AMPI_SENDRECVREPLACE","AMPI_SCATTER","AMPI_GATHER","AMPI_SCATTERV","AMPI_GATHERV","AMPI_SEND_INIT","AMPI_RECV_INIT","AMPI_START","AMPI_STARTALL","AMPI_SENDRECV","AMPI_BRECV","AMPI_ALLGATHER","AMPI_ALLGATHERV"};
#endif

/**@}*/

#include <stdlib.h>
#include <assert.h>

#include <ampi.h>

/*int ampi_vac=0;*/

/** 
 * An element of the AMPI tape. 
 */
typedef struct ampi_tape_entry {
    int oc; /**< Operation code */
    int *arg; /**< Array of arrguments */
    INT64 *idx; /**< AD tool tape indices */
    ampi_stack* stack; /**< Stack for stored primal values */
    AMPI_Request_t *request; /**< Saved pointer to an active AMPI_Request */
    MPI_Comm comm; /**< MPI_Communicator */
    int tag; /**< MPI_Tag */
} ampi_tape_entry;

/*! AMPI taping routines which have an MPI counterpart that is adjoined.*/

/**
 * Initialize AMPI consisting of allocating the AMPI tape and calling AMPI_Init_f
 *
 * @param argc Forwarded to MPI. 
 * @param argv Forwarded to MPI
 *
 * @return error code
 */
int AMPI_Init(int* argc, char*** argv);

/**
 * AMPI Finalize is called _after_ the reverse section. Hence AMPI_Init_b is
 * called here.
 *
 * @return error code
 */
int AMPI_Finalize();

/**
 * @brief Active blocking send with active buffer
 *
 * @param buf Pointer to active buffer
 *
 */
int AMPI_Send(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

/**
 * @brief Active blocking buffered send with active buffer
 *
 * @param buf Pointer to active buffer
 *
 */
int AMPI_Bsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

/**
 * @brief Active blocking receive with active buffer
 *
 * @param buf Pointer to active buffer
 *
 */
int AMPI_Recv(void* buf, int count, MPI_Datatype datatype, int src, int tag, MPI_Comm comm, MPI_Status* status);

/**
 * @brief Active non blocking send with active buffer
 *
 * @param buf Pointer to active buffer
 */
int AMPI_Isend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, AMPI_Request* request);

/**
 * @brief Active non blocking receive with active buffer
 *
 * @param buf Pointer to active buffer
 *
 */
int AMPI_Irecv(void* buf, int count, MPI_Datatype datatype, int src, int tag, MPI_Comm comm, AMPI_Request* request);

/**
 * @brief Active wait. If the MPI_Request is in the hash table, the communication is active
 *
 */
int AMPI_Wait(AMPI_Request *, MPI_Status *);

/**
 * @brief Active waitall. If the MPI_Request is in the hash table, the
 * communication is active, otherwise only MPI_Wait is called.
 *
 */
int AMPI_Waitall(int , AMPI_Request *, MPI_Status *);

/**
 * @brief Active waitall. If the MPI_Request is in the hash table, the
 * communication is active, otherwise only MPI_Wait is called. If no requests
 * are left, MPI_Waitany is called one last time to set the MPI library's
 * handlers correctly.
 *
 */
int AMPI_Waitany(int count, AMPI_Request array_of_requests[], int *index, MPI_Status *status);

/**
 * @brief Passthrough to MPI_Test, unwrapping the MPI_Request from AMPI_Request.
 *
 */
int AMPI_Test(AMPI_Request * request, int *flag, MPI_Status * status);

/**
 * @brief Active broadcast. 
 *
 * @param buf Pointer to active buffer
 */
int AMPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

/**
 * @brief Active reduce. 
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

/**
 * @brief Active allreduce. 
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

/**
 * @brief Active scatter. 
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Scatter(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);

/**
 * @brief Active scatterv.
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);

/**
 * @brief Active gather. 
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Gather(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);

/**
 * @brief Active allgather.
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Allgather(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, MPI_Comm comm);

/**
 * @brief Active gatherv.
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Gatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);

/**
 * @brief Active allgatherv.
 *
 * @param sendbuf Pointer to active send buffer
 * @param recvbuf Pointer to active receive buffer
 */
int AMPI_Allgatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);

/**
 * @brief Active receive init. The active AMPI_Requests
 * are registered. MPI_Recv_init is _not_ executed in AMPI. See AMPI_Start.
 *
 * @param buf Pointer to active receive buffer
 */
int AMPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, AMPI_Request *request);

/**
 * @brief Active receive init. The active AMPI_Requests
 * are registered. MPI_Send_init is _not_ executed in AMPI. See AMPI_Start.
 *
 * @param buf Pointer to active send buffer
 */
int AMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, AMPI_Request *request);

/** @brief Active start. Call AMPI_Isend or AMPI_Irecv for active request
 *
 */
int AMPI_Start(AMPI_Request *request);

/** @brief Active startall. Call AMPI_Start count times
 *
 */
int AMPI_Startall(int count, AMPI_Request array_of_requests[]);

/** @brief Active sendrecv with replace. 
 *
 * @param buf Pointer to active buf
 */
int AMPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status);

int AMPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
/** Release a tape handle */
void ampi_release_tape(ampi_tape_entry* ampi_tape);

/** Create a tape handle*/
ampi_tape_entry* ampi_create_tape(long int size);

#endif
