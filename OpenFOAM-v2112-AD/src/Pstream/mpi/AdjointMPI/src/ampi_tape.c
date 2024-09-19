/* Generic AMPI C tape. Don't touch this. The code is always mirrored with the AMPI repo.
 * Changing code here will result in merge conflicts.
 *
 * See header for more information.
 */
#define NO_COMM_WORLD

#include <stddef.h>
#include <ampi_tape.h>
#ifdef __cplusplus
#include <ampi_interface.hpp>
#endif
#ifdef AMPI_COUNT_COMMS
int ampi_comm_count=0;
#endif

//#define AMPI_DEBUG

ampi_tape_entry *ampi_tape;

int AMPI_Init(int* argc, char*** argv) {
    return AMPI_Init_f(argc, argv);
}

int AMPI_Finalize() {
#ifdef AMPI_COUNT_COMMS
    printf("AMPI comunications executed: %d\n", ampi_comm_count);
#endif
    return AMPI_Init_b(NULL, NULL);

}

void ampi_create_dummies(void *buf, int *size) {
    int displ = 0;
    ampi_create_dummies_displ(buf, &displ, size);
}

int AMPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    if(datatype!=AMPI_DOUBLE) {
      return MPI_Send(buf, count, datatype, dest, tag, comm);
    }
    int i=0;
    double *primalValues = (double*) malloc(sizeof(double)*count);

    for(i=0;i<count;i=i+1) {
        ampi_get_val(buf,&i,&primalValues[i]);
    }

    if (ampi_isTapeActive()){

      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);

      ampi_tape->arg=(int*) malloc(sizeof(int)*2);

      ampi_create_tape_entry((void*)ampi_tape);

      for(i=0;i<count;i=i+1) {
        ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }

      /*create actual MPI entry*/

      ampi_tape->oc = AMPI_SEND;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;
    }

    int exitCode = AMPI_Send_f(primalValues, count, datatype, dest, tag, comm);

    free(primalValues);

    return exitCode;
}

int AMPI_Bsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    if(datatype!=AMPI_DOUBLE) {
      return MPI_Bsend(buf, count, datatype, dest, tag, comm);
    }
    int i=0;
    /*Allocate one more element and hint that it is a Bsend*/
    double *primalValues = (double*) malloc(sizeof(double)*(count+1));
    primalValues[count]=1;

    for(i=0;i<count;i=i+1) {
        ampi_get_val(buf,&i,&primalValues[i]);
    }

    if (ampi_isTapeActive()){

      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);

      ampi_tape->arg=(int*) malloc(sizeof(int)*2);

      ampi_create_tape_entry((void*)ampi_tape);
      for(i=0;i<count;i=i+1) {
        ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }

      /*create actual MPI entry*/

      /*BSEND is adjoined like a AMPI_SEND. The corresponding AMPI_RECV is adjoined as a BSEND!*/
      ampi_tape->oc = AMPI_SEND;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;
    }

    int exitCode = AMPI_Bsend_f(primalValues, count+1, datatype, dest, tag, comm);

    free(primalValues);

    return exitCode;
}

int AMPI_Recv(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Status *status) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    if(datatype!=AMPI_DOUBLE) {
      return MPI_Recv(buf, count, datatype, dest, tag, comm, status);
    }

    int i=0;
    /*One element longer to distinguish between Bsend and Send of the received message*/
    double *primalValues = (double*) malloc(sizeof(double)*(count+1));
    /* set to 0. if this is a 1, the received message was a Bsend */
    primalValues[count]=0;

    int exitCode = AMPI_Recv_f(primalValues, count+1, datatype, dest, tag, comm, status);
    /*if 1, it is a BSEND on the other end*/
    if (ampi_isTapeActive()){
      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);
      ampi_tape->arg=(int*) malloc(sizeof(int)*2);

      ampi_create_dummies(buf, &count);

      ampi_create_tape_entry((void*)ampi_tape);

      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->comm = comm;
      if(status==MPI_STATUS_IGNORE) {
        ampi_tape->tag = tag;
      } else {
        ampi_tape->tag = status->MPI_TAG;
      }

      for(i=0;i<count;i=i+1) {
        ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }
      if(primalValues[count]==1) {
        ampi_tape->oc = AMPI_BRECV;
      }
      else {
        ampi_tape->oc = AMPI_RECV;
      }
    }

    for(i=0;i<count;i=i+1) {
        ampi_set_val(buf, &i, &primalValues[i]);
    }
    free(primalValues);
    return exitCode;
}

int AMPI_Isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, AMPI_Request *mpi_request_p) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

    AMPI_CREATE_REQUEST(mpi_request, mpi_request_p);

    if(datatype!=AMPI_DOUBLE) {
      return MPI_Isend(buf, count, datatype, dest, tag, comm, &mpi_request->mpiRequest);
    }
    int i=0, temp;
    double *primalValues = (double*) malloc(sizeof(double)*count);

    for(i=0 ; i<count ; i=i+1) {
      ampi_get_val(buf,&i,&primalValues[i]);
    }

    mpi_request->buf = buf;

    if (ampi_isTapeActive()){

      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);
      ampi_tape->arg=(int*) malloc(sizeof(int)*2);

      /*create dummy of each element*/

      for(i=0 ; i<count ; i=i+1) {
        ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }
      ampi_tape->oc = AMPI_ISEND;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;

      ampi_create_tape_entry((void*)ampi_tape);
      ampi_tape->request = (AMPI_Request_t*) malloc(sizeof(AMPI_Request_t));
      mpi_request->va = ampi_tape;

    }

    /*point current primalRequest index to this tape entry*/
    mpi_request->v = primalValues;
    temp = AMPI_Isend_f(primalValues, count, datatype, dest, tag, comm, mpi_request_p);

    return temp;
}

int AMPI_Irecv(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, AMPI_Request *mpi_request_p) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

    AMPI_CREATE_REQUEST(mpi_request, mpi_request_p);

    if(datatype!=AMPI_DOUBLE) {
      return MPI_Irecv(buf, count, datatype, dest, tag, comm, &mpi_request->mpiRequest);
    }
    int i=0;    
    /* One more element. Could be a Bsend on the other side. This is only
     * important to avoid deadlocks in the blocking case. Irrelevant in this
     * case, however the element needs to be there */
    double * tmp = (double*) malloc(sizeof(double)*(count+1));
    tmp[count]=0;

    mpi_request->buf = buf;

    if (ampi_isTapeActive()){
      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);
      ampi_tape->arg=(int*) malloc(sizeof(int)*2);

      ampi_create_tape_entry((void*)ampi_tape);

      ampi_create_dummies(buf, &count);

      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->oc = AMPI_IRECV;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;

      for(i=0 ; i<count ; i=i+1) {
        ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }
      ampi_tape->request = (AMPI_Request_t*) malloc(sizeof(AMPI_Request_t));
      mpi_request->va = ampi_tape;
    }

    int temp = AMPI_Irecv_f(tmp, count+1, datatype, dest, tag, comm, mpi_request_p);

    return temp;
}

int AMPI_Wait(AMPI_Request *mpi_request_p, MPI_Status *status) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

  if(AMPI_REQUEST_NULL == *mpi_request_p) {
    return 0;
  }

  AMPI_GET_REQUEST(mpi_request, mpi_request_p);

  int result = MPI_Wait(&mpi_request->mpiRequest, status);
  int i = 0;
  //int ret = 0;
  if (NULL != mpi_request->va) {

    double *primalValues = (double *) mpi_request->v;
    /*get the corresponding isend or irecv tape entry*/

    /*finally copy the primalRequest to the tape*/
    if (status == MPI_STATUS_IGNORE) {
      mpi_request->tag = mpi_request->tag;
    } else {
      mpi_request->tag = status->MPI_TAG;
    }

    if (mpi_request->oc == AMPI_IR) {
      for (i = 0; i < mpi_request->size; i = i + 1) {
        ampi_set_val(mpi_request->buf, &i, &primalValues[i]);
      }
    }

    if (ampi_isTapeActive()){
      ampi_tape_entry *ampi_tape = ampi_create_tape(1);
      ampi_tape->oc = AMPI_WAIT;
      ampi_tape->tag=mpi_request->tag;
      ampi_tape->request = mpi_request->va->request;
      *mpi_request->va->request = *mpi_request;
      /*ampi_tape->primalRequest->a = &ampi_tape[primalRequest->va].d;*/
      ampi_tape->request->size = mpi_request->va->arg[0];
      ampi_tape->request->va = mpi_request->va; /* link here the information backward */
      ampi_create_tape_entry((void *) ampi_tape);
    }
    free(primalValues);
  }

  AMPI_DELETE_REQUEST(mpi_request, mpi_request_p);

  return result;
}

int AMPI_Waitall(int count, AMPI_Request *mpi_request, MPI_Status *status) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    int i=0;
    int exitCode=-1;
    if (status != MPI_STATUSES_IGNORE) {
        for(i=0; i<count; i=i+1) {
            exitCode=AMPI_Wait(&mpi_request[i],&status[i]);
        }
    } else {
        for(i=0; i<count; i=i+1) {
            exitCode=AMPI_Wait(&mpi_request[i],MPI_STATUS_IGNORE);
        }
    }
    return exitCode;
}
int AMPI_Waitany(int count, AMPI_Request array_of_requests[], int *index, MPI_Status *status) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    int exitCode=AMPI_Waitany_f(count,array_of_requests,index,status);
    if(*index < count && *index >=0) {

      AMPI_GET_REQUEST(mpi_request, &array_of_requests[*index]);
      int i=0;
      if (NULL != mpi_request->va) {

        double *primalValues = (double *) mpi_request->v;
        /*get the corresponding isend or irecv tape entry*/
        /*finally copy the primalRequest to the tape*/
        if (status == MPI_STATUS_IGNORE) {
          mpi_request->tag = mpi_request->tag;
        } else {
          mpi_request->tag = status->MPI_TAG;
        }

        if (mpi_request->oc == AMPI_IR) {
          for (i = 0; i < mpi_request->size; i = i + 1) {
            ampi_set_val(mpi_request->buf, &i, &primalValues[i]);
          }
        }

        if (ampi_isTapeActive()){
          ampi_tape_entry *ampi_tape = ampi_create_tape(1);
          ampi_tape->oc = AMPI_WAIT;
          ampi_tape->tag=mpi_request->tag;
          ampi_tape->request = mpi_request->va->request;
          *mpi_request->va->request = *mpi_request;
          /*ampi_tape->primalRequest->a = &ampi_tape[primalRequest->va].d;*/
          ampi_tape->request->size = mpi_request->va->arg[0];
          ampi_tape->request->va = mpi_request->va; /* link here the information backward */
          ampi_create_tape_entry((void *) ampi_tape);
        }

        free(primalValues);
      }

      AMPI_DELETE_REQUEST(mpi_request, &array_of_requests[*index]);
    }
    return exitCode;
}

int AMPI_Test(AMPI_Request * request, int *flag, MPI_Status * status){
    return MPI_Test(&((*request)->mpiRequest), flag, status);
}

int AMPI_Bcast(void *buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    if(datatype!=AMPI_DOUBLE) {
      return MPI_Bcast(buf, count, datatype, root, comm);
    }
    int rank=0;
    int i=0;
    double *primalValues = (double*) malloc(sizeof(double)*count);
    for(i=0;i<count;i++) primalValues[i]=0;
    MPI_Comm_rank(comm,&rank);

    if(rank==root) {
      for(i = 0 ; i < count ; i=i+1) {
        ampi_get_val(buf,&i,&primalValues[i]);
      }
    }

    if (ampi_isTapeActive()){

      ampi_tape_entry* ampi_tape = ampi_create_tape(count+1);

      if(rank==root) {
        for(i = 0 ; i < count ; i=i+1) {
          ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
        }
      }

      /*create actual MPI entry*/
      if(rank!=root) {
        ampi_create_dummies(buf, &count);
      }

      ampi_tape->arg=(int*) malloc(sizeof(int)*2);
      ampi_tape->oc = AMPI_BCAST;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = root;
      ampi_tape->comm = comm;

      if(rank!=root) {
        for(i=0;i<count;i=i+1) {
          ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
        }
      }
      ampi_create_tape_entry((void*)ampi_tape);
    }

    int temp = AMPI_Bcast_f(primalValues, count, datatype, root, comm);

    if(rank!=root) {
      for(i=0;i<count;i=i+1) {
          ampi_set_val(buf, &i, &primalValues[i]);
        }
      }

    free(primalValues);

    return temp;
}

int AMPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    if(datatype!=AMPI_DOUBLE) {
      return MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
    }
    int i=0;
    int rank=0;
    MPI_Comm_rank(comm,&rank);
    double* tmp_send = NULL;
    double* tmp_recv = NULL;
    if(root == rank) {
      tmp_recv = (double*) malloc(sizeof(double)*count);
    }

    if(root == rank && sendbuf == MPI_IN_PLACE) {
      tmp_send = MPI_IN_PLACE;
    }
    else {
      tmp_send = (double*) malloc(sizeof(double)*count);
    }

    for(i=0 ; i<count ; i=i+1) {
        if(sendbuf != MPI_IN_PLACE) {
          ampi_get_val(sendbuf,&i,&tmp_send[i]);
        }
        if(root == rank) {
          ampi_get_val(recvbuf,&i,&tmp_recv[i]);
        }
    }

    if (ampi_isTapeActive()){
      ampi_tape_entry* ampi_tape = ampi_create_tape(2*count+1);


      ampi_tape->arg=(int*) malloc(sizeof(int)*3);

      if(root == rank && sendbuf == MPI_IN_PLACE) {
        for(i=0 ; i<count ; i=i+1) {
            ampi_get_idx(recvbuf, &i, &ampi_tape->idx[i]);
        }
      } else {
        for(i=0 ; i<count ; i=i+1) {
            ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
        }
      }

      /*sendbuf dummies*/
      if(rank == root) {
        ampi_create_dummies(recvbuf, &count);
      }

      ampi_create_tape_entry((void*)ampi_tape);


      /*actual reduce entry*/

      ampi_tape->oc = AMPI_REDUCE;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = root;
      if(op == MPI_SUM){
          ampi_tape->arg[2] = AMPI_REDUCE_ADD;
      }
      if(op == MPI_PROD){
          ampi_tape->arg[2] = AMPI_REDUCE_MUL;
      }
      if(op == MPI_MIN){
          ampi_tape->arg[2] = AMPI_REDUCE_MIN;
      }
      if(op == MPI_MAX){
          ampi_tape->arg[2] = AMPI_REDUCE_MAX;
      }
      ampi_tape->comm = comm;

      AMPI_Reduce_f(tmp_send, tmp_recv, count, datatype, op, root, comm, &ampi_tape->stack);

      /*recvbuf entry*/

      if(rank == root) {
        for(i=0 ; i<count ; i=i+1) {
            ampi_get_idx(recvbuf, &i, &ampi_tape->idx[count + i]);
        }
      }

    }else{
       MPI_Reduce(tmp_send, tmp_recv, count, datatype, op, root, comm);
    }


    if(rank == root) {
      for(i=0 ; i<count ; i=i+1) {
          ampi_set_val(recvbuf, &i, &tmp_recv[i]);
      }
    }

    if(tmp_send != MPI_IN_PLACE) {
      free(tmp_send);
    }
    if(rank == root) {
      free(tmp_recv);
    }
    return 0;
}

int AMPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  int ierr=0;
#ifdef AMPI_COUNT_COMMS
  ampi_comm_count=ampi_comm_count+1;
#endif
  if(datatype!=AMPI_DOUBLE) {
    return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
  }

  int i=0;
  double * tmp_send = MPI_IN_PLACE;
  if(sendbuf != MPI_IN_PLACE) {
    tmp_send = (double*) malloc(sizeof(double)*count);
  }
  double * tmp_recv = (double*) malloc(sizeof(double)*count);

  if(sendbuf == MPI_IN_PLACE) {
    for(i=0 ; i<count ; i=i+1) {
      ampi_get_val(recvbuf,&i,&tmp_recv[i]);
    }
  } else {
    for(i=0 ; i<count ; i=i+1) {
      ampi_get_val(sendbuf,&i,&tmp_send[i]);
    }
  }

  if(ampi_isTapeActive()) {

    ampi_tape_entry* ampi_tape = ampi_create_tape(2*count+1);

    ampi_tape->arg=(int*) malloc(sizeof(int)*3);

    if(sendbuf == MPI_IN_PLACE) {
      for(i=0 ; i<count ; i=i+1) {
        ampi_get_idx(recvbuf, &i, &ampi_tape->idx[i]);
      }
    } else {
      for(i=0 ; i<count ; i=i+1) {
        ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
      }
    }

    /*sendbuf dummies*/
    ampi_create_dummies(recvbuf, &count);
    ampi_create_tape_entry((void*)ampi_tape);;
    /*actual reduce entry*/

    ampi_tape->oc = AMPI_ALLREDUCE;
    ampi_tape->arg[0] = count;
    ampi_tape->comm = comm;
    /*ampi_tape->arg[1] = root;*/
    if(op == MPI_SUM){
      ampi_tape->arg[2] = AMPI_REDUCE_ADD;
    }
    if(op == MPI_PROD){
      ampi_tape->arg[2] = AMPI_REDUCE_MUL;
    }
    if(op == MPI_MIN){
      ampi_tape->arg[2] = AMPI_REDUCE_MIN;
    }
    if(op == MPI_MAX){
      ampi_tape->arg[2] = AMPI_REDUCE_MAX;
    }

    ierr=AMPI_Allreduce_f(tmp_send, tmp_recv, count, datatype, op, comm, &ampi_tape->stack);

    for(i=0 ; i<count ; i=i+1) {
      ampi_get_idx(recvbuf, &i, &ampi_tape->idx[count + i]);
    }

  }else{
    ierr = MPI_Allreduce(tmp_send, tmp_recv, count, datatype, op, comm);
  }


  for(i=0 ; i<count ; i=i+1) {
    ampi_set_val(recvbuf, &i, &tmp_recv[i]);
  }


  if(sendbuf != MPI_IN_PLACE) {
    free(tmp_send);
  }
  free(tmp_recv);
  return ierr;
}

int AMPI_Scatter(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Scatter(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm);
  }
  int i=0;
  int size=0;
  int rank=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  double * tmp_send = NULL;
  double * tmp_recv = (double*) malloc(sizeof(double)*recvcnt);

  int sendSize=0;
  int bufferSize = recvcnt+1;
  if(rank == root) {
    sendSize=sendcnt*size;
    bufferSize = sendSize+recvcnt+1;

    tmp_send = (double*) malloc(sizeof(double)*sendSize);

    for(i=0 ; i<sendSize ; i=i+1) {
      ampi_get_val(sendbuf,&i,&tmp_send[i]);
    }
  }

  if (ampi_isTapeActive()){

    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);
    ampi_tape->arg=(int*) malloc(sizeof(int)*3);
    ampi_tape->comm = comm;
    ampi_create_dummies(recvbuf, &recvcnt);

    /*sendbuf dummies*/

    if(rank == root) {
      for(i=0 ; i<sendSize ; i=i+1) {
        ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
      }
    }

    ampi_tape->oc = AMPI_SCATTER;
    ampi_tape->arg[2] = sendcnt;
    ampi_tape->arg[1] = recvcnt;
    ampi_tape->arg[0] = root;
    ampi_tape->comm = comm;
    ampi_create_tape_entry((void*)ampi_tape);;
    /*recvbuf entry*/

    for(i=0 ; i<recvcnt ; i=i+1) {
      ampi_get_idx(recvbuf, &i, &ampi_tape->idx[sendSize+i]);
    }
  }

  AMPI_Scatter_f(tmp_send, sendcnt, sendtype, tmp_recv, recvcnt, recvtype, root, comm);

  for(i=0 ; i<recvcnt ; i=i+1) {
    ampi_set_val(recvbuf, &i, &tmp_recv[i]);
  }

  if(rank == root) {
    free(tmp_send);
  }
  free(tmp_recv);
  return 0;

}

int AMPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Scatterv(sendbuf, sendcnts, displs, sendtype, recvbuf, recvcnt, recvtype, root, comm);
  }
  int i=0;
  int j=0;
  int size=0;
  int rank=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  double * tmp_send = NULL;
  int *    tmp_disp = NULL;
  double * tmp_recv = (double*) malloc(sizeof(double)*recvcnt);

  int sendSize=0;
  int bufferSize = recvcnt+1;
  if(rank == root) {
    tmp_disp = (int*) malloc(sizeof(int) * size);
    for(i=0;i<size;i=i+1) {
      tmp_disp[i] = sendSize;

      sendSize = sendSize + sendcnts[i];
    }
    bufferSize = sendSize+recvcnt+1;

    tmp_send = (double*) malloc(sizeof(double)*sendSize);

    int idxPos = 0;
    for(i=0 ; i<size ; i=i+1) {
      for(j=0 ; j<sendcnts[i]; j=j+1) {
        int pos = displs[i] + j;
        ampi_get_val(sendbuf,&pos,&tmp_send[idxPos]);
        idxPos=idxPos+1;
      }
    }
  }

  if (ampi_isTapeActive()){

    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);
    ampi_tape->arg=(int*) malloc(sizeof(int)*3 + 2*size);
    ampi_tape->comm = comm;
    ampi_create_dummies(recvbuf, &recvcnt);

    /*sendbuf dummies*/

    if(rank == root) {
      for(i=0 ; i<sendSize ; i=i+1) {
        ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
      }
    }

    ampi_tape->oc = AMPI_SCATTERV;
    for(i=0 ; i<size; i=i+1) {
      ampi_tape->arg[3 + i] = sendcnts[i];
      ampi_tape->arg[3 + size + i] = tmp_disp[i];
    }
    ampi_tape->arg[2] = sendSize;
    ampi_tape->arg[1] = recvcnt;
    ampi_tape->arg[0] = root;
    ampi_tape->comm = comm;
    ampi_create_tape_entry((void*)ampi_tape);;
    /*recvbuf entry*/

    for(i=0 ; i<recvcnt ; i=i+1) {
      ampi_get_idx(recvbuf, &i, &ampi_tape->idx[sendSize+i]);
    }
  }

  MPI_Scatterv(tmp_send, sendcnts, tmp_disp, sendtype, tmp_recv, recvcnt, recvtype, root, comm);

  for(i=0 ; i<recvcnt ; i=i+1) {
    ampi_set_val(recvbuf, &i, &tmp_recv[i]);
  }

  if(rank == root) {
    free(tmp_disp);
    free(tmp_send);
  }
  free(tmp_recv);
  return 0;

}

int AMPI_Gather(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Gather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root, comm);
  }
  int i=0;
  int size=0;
  int rank=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  double * tmp_send = (double*)malloc(sizeof(double)*sendcnt);
  double * tmp_recv = 0;
  if(rank == root) {
    tmp_recv = (double*)malloc(sizeof(double)*recvcnt*size);
  }

  /* check for size */
  int recvSize=recvcnt*size;
  int bufferSize = sendcnt+1;
  if(rank == root) {
    bufferSize = recvcnt*size+sendcnt+1;
  }

  for(i=0 ; i<sendcnt ; i=i+1) {
    ampi_get_val(sendbuf,&i,&tmp_send[i]);
  }

  if (ampi_isTapeActive()){
    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);

    ampi_tape->arg=(int*)malloc(sizeof(int)*3);
    ampi_tape->comm = comm;


    if(rank == root) {
      ampi_create_dummies(recvbuf, &recvSize);
    }

    /*sendbuf dummies*/

    for(i=0 ; i<sendcnt ; i=i+1) {
      ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
    }

    ampi_tape->oc = AMPI_GATHER;
    ampi_tape->arg[2] = sendcnt;
    ampi_tape->arg[1] = recvcnt;
    ampi_tape->arg[0] = root;
    ampi_tape->comm = comm;
    ampi_create_tape_entry((void*)ampi_tape);;

    /*recvbuf entry*/

    if(rank == root) {
      for(i=0 ; i<recvcnt*size ; i=i+1) {
        ampi_get_idx(recvbuf, &i, &ampi_tape->idx[sendcnt + i]);
      }
    }
  }

  AMPI_Gather_f(tmp_send, sendcnt, sendtype, tmp_recv, recvcnt, recvtype, root, comm);

  if(rank == root) {
    for(i=0 ; i<recvcnt*size ; i=i+1) {
      ampi_set_val(recvbuf, &i, &tmp_recv[i]);
    }
  }
  free(tmp_send);
  if( rank == root ) {
    free(tmp_recv);
  }
  return 0;

}

int AMPI_Gatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Gatherv(sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, root, comm);
  }

  int i=0;
  int size=0;
  int rank=0;
  int total_size=0;
  int bufferSize=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_size(comm,&rank);

  /* Determine and allocate maximum size of recvbuf */

  bufferSize = sendcnt + 1;
  double * tmp_send = (double*) malloc(sizeof(double)*sendcnt);
  int *    tmp_disp = (int*) malloc(sizeof(int) * size);
  double * tmp_recv = NULL;

  for(i=0 ; i<sendcnt ; i=i+1) {
    ampi_get_val(sendbuf,&i,&tmp_send[i]);
  }

  if(rank == root) {
    for(i=0;i<size;i=i+1) {
      tmp_disp[i] = total_size;

      total_size = total_size + recvcnts[i];
    }

    bufferSize = bufferSize + total_size;
    tmp_recv = (double*) malloc(sizeof(double)*total_size);
  }

  if (ampi_isTapeActive()){
    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);
    ampi_tape->arg=(int*) malloc(sizeof(int)*(3 + 2*size));
    ampi_tape->comm = comm;

    if(rank == root) {
      for( i=0; i < size; i=i+1 ) {
        ampi_create_dummies_displ(recvbuf, &displs[i], &recvcnts[i]);
      }
    }

    /*sendbuf dummies*/

    for(i=0 ; i<sendcnt ; i=i+1) {
      ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
    }
    ampi_tape->oc = AMPI_GATHERV;
    for(i=0 ; i<size; i=i+1) {
      ampi_tape->arg[3 + i] = recvcnts[i];
      ampi_tape->arg[3 + size + i] = tmp_disp[i];
    }
    ampi_tape->arg[2] = total_size;
    ampi_tape->arg[1] = sendcnt;
    ampi_tape->arg[0] = root;
    ampi_tape->comm = comm;


    /*recvbuf entry*/

    if(rank == root) {
      int j;
      int idxPos = sendcnt;
      for( i=0; i < size; i=i+1 ) {
        for( j=0; j < recvcnts[i]; j=j+1) {
          int pos = displs[i] + j;
          ampi_get_idx(recvbuf, &pos, &ampi_tape->idx[idxPos]);
          idxPos=idxPos+1;
        }
      }
    }
  }

  MPI_Gatherv(tmp_send, sendcnt, sendtype, tmp_recv, recvcnts, tmp_disp, recvtype, root, comm);

  if(rank == root) {
    int j;
    int idxPos = 0;
    for( i=0; i < size; i=i+1 ) {
      for( j=0; j < recvcnts[i]; j=j+1) {
        int pos = displs[i] + j;
        ampi_set_val(recvbuf, &pos, &tmp_recv[idxPos]);
        idxPos=idxPos+1;
      }
    }
  }

  free(tmp_send);
  free(tmp_disp);
  if(rank == root) {
    free(tmp_recv);
  }

  return 0;
}

int AMPI_Allgather(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Allgather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, comm);
  }
  int i=0;
  int size=0;
  int rank=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  double * tmp_send = (double*)malloc(sizeof(double)*sendcnt);
  double * tmp_recv = (double*)malloc(sizeof(double)*recvcnt*size);

  /* check for size */
  int recvSize=recvcnt*size;
  int bufferSize = recvcnt*size+sendcnt+1;

  for(i=0 ; i<sendcnt ; i=i+1) {
    ampi_get_val(sendbuf,&i,&tmp_send[i]);
  }

  if (ampi_isTapeActive()){
    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);

    ampi_tape->arg=(int*)malloc(sizeof(int)*2);
    ampi_tape->comm = comm;

    ampi_create_dummies(recvbuf, &recvSize);

    /*sendbuf dummies*/

    for(i=0 ; i<sendcnt ; i=i+1) {
      ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
    }

    ampi_tape->oc = AMPI_ALLGATHER;
    ampi_tape->arg[1] = sendcnt;
    ampi_tape->arg[0] = recvcnt;
    ampi_tape->comm = comm;
    ampi_create_tape_entry((void*)ampi_tape);;

    /*recvbuf entry*/

    for(i=0 ; i<recvcnt*size ; i=i+1) {
      ampi_get_idx(recvbuf, &i, &ampi_tape->idx[sendcnt + i]);
    }
  }

  MPI_Allgather(tmp_send, sendcnt, sendtype, tmp_recv, recvcnt, recvtype, comm);

  for(i=0 ; i<recvcnt*size ; i=i+1) {
    ampi_set_val(recvbuf, &i, &tmp_recv[i]);
  }
  free(tmp_send);
  free(tmp_recv);

  return 0;
}

int AMPI_Allgatherv(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int *recvcnts, int *displs, MPI_Datatype recvtype, MPI_Comm comm) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(sendtype !=AMPI_DOUBLE || recvtype != AMPI_DOUBLE) {
    return MPI_Allgatherv(sendbuf, sendcnt, sendtype, recvbuf, recvcnts, displs, recvtype, comm);
  }

  int i=0;
  int size=0;
  int rank=0;
  int total_size=0;
  int bufferSize=0;
  MPI_Comm_size(comm,&size);
  MPI_Comm_size(comm,&rank);

  /* Determine and allocate maximum size of recvbuf */

  double * tmp_send = (double*) malloc(sizeof(double)*sendcnt);
  int *    tmp_disp = (int*) malloc(sizeof(int) * size);

  for(i=0 ; i<sendcnt ; i=i+1) {
    ampi_get_val(sendbuf,&i,&tmp_send[i]);
  }

  for(i=0;i<size;i=i+1) {
    tmp_disp[i] = total_size;

    total_size = total_size + recvcnts[i];
  }

  bufferSize = sendcnt + 1 + total_size;
  double * tmp_recv = (double*) malloc(sizeof(double)*total_size);

  if (ampi_isTapeActive()){
    ampi_tape_entry* ampi_tape = ampi_create_tape(bufferSize);
    ampi_tape->arg=(int*) malloc(sizeof(int)*(1 + 4*size));
    ampi_tape->comm = comm;

    for( i=0; i < size; i=i+1 ) {
      ampi_create_dummies_displ(recvbuf, &displs[i], &recvcnts[i]);
    }

    /*sendbuf dummies*/

    for(i=0 ; i<sendcnt ; i=i+1) {
      ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
    }
    ampi_tape->oc = AMPI_ALLGATHERV;
    for(i=0 ; i<size; i=i+1) {
      ampi_tape->arg[1 +          i] = recvcnts[i];
      ampi_tape->arg[1 +   size + i] = tmp_disp[i];
      /* prepare adjoint send information for the adjoint call */
      ampi_tape->arg[1 + 2*size + i] = sendcnt;
      ampi_tape->arg[1 + 3*size + i] = i * sendcnt;
    }
    ampi_tape->arg[0] = total_size;
    ampi_tape->comm = comm;


    /*recvbuf entry*/

    int j;
    int idxPos = sendcnt;
    for( i=0; i < size; i=i+1 ) {
      for( j=0; j < recvcnts[i]; j=j+1) {
        int pos = displs[i] + j;
        ampi_get_idx(recvbuf, &pos, &ampi_tape->idx[idxPos]);
        idxPos=idxPos+1;
      }
    }
  }

  MPI_Allgatherv(tmp_send, sendcnt, sendtype, tmp_recv, recvcnts, tmp_disp, recvtype, comm);

  int j;
  int idxPos = 0;
  for( i=0; i < size; i=i+1 ) {
    for( j=0; j < recvcnts[i]; j=j+1) {
      int pos = displs[i] + j;
      ampi_set_val(recvbuf, &pos, &tmp_recv[idxPos]);
    }
  }
  free(tmp_send);
  free(tmp_disp);
  free(tmp_recv);
  return 0;
}

int AMPI_Recv_init(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, AMPI_Request *mpi_request_p) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

  AMPI_CREATE_REQUEST(mpi_request, mpi_request_p);

  if(datatype!=AMPI_DOUBLE) {
    return MPI_Recv_init(buf, count, datatype, source, tag, comm, &mpi_request->mpiRequest);
  }
    mpi_request->buf = buf;

    if (ampi_isTapeActive()){
      ampi_tape_entry* ampi_tape = ampi_create_tape(1);
      ampi_tape->arg=(int*) malloc(sizeof(int)*2);
      mpi_request->va=ampi_tape;

      ampi_create_tape_entry((void*)ampi_tape);;
      ampi_tape->oc = AMPI_RECV_INIT;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = source;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;
    }

    return 0;
}
int AMPI_Send_init(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, AMPI_Request *mpi_request_p) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

  AMPI_CREATE_REQUEST(mpi_request, mpi_request_p);

  if(datatype!=AMPI_DOUBLE) {
    return MPI_Send_init(buf, count, datatype, dest, tag, comm, &mpi_request->mpiRequest);
  }
    mpi_request->buf = buf;

    if (ampi_isTapeActive()){
      ampi_tape_entry* ampi_tape = ampi_create_tape(1);
      ampi_tape->arg=(int*) malloc(sizeof(int)*2);
      mpi_request->va=ampi_tape;

      ampi_create_tape_entry((void*)ampi_tape);;
      ampi_tape->oc = AMPI_SEND_INIT;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->comm = comm;
      ampi_tape->tag = tag;
    }
    return 0;
}
int AMPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(datatype!=AMPI_DOUBLE) {
    return MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag, comm, status);
  }
    int i=0;
    double * tmp = (double*) malloc(sizeof(double)*count);
    for(i=0;i<count;i=i+1) {
        ampi_get_val(buf,&i,&tmp[i]);
    }

    if (ampi_isTapeActive()){
      /*create actual MPI entry*/
      ampi_tape_entry* ampi_tape = ampi_create_tape(2*count+2);
      ampi_tape->arg=(int*) malloc(sizeof(int)*4);

      for(i=0;i<count;i=i+1) {
          ampi_get_idx(buf, &i, &ampi_tape->idx[i]);
      }
      ampi_tape->oc = AMPI_SENDRECVREPLACE;
      ampi_tape->arg[0] = count;
      ampi_tape->arg[1] = dest;
      ampi_tape->arg[2] = source;

      ampi_tape->comm = comm;
      ampi_tape->tag = sendtag;

      ampi_create_dummies(buf, &count);
      ampi_create_tape_entry((void*)ampi_tape);;


      if(status!=MPI_STATUS_IGNORE) {
        ampi_tape->arg[3] = status->MPI_TAG;
      } else {
        ampi_tape->arg[3] = recvtag;
      }

      for(i=0;i<count;i=i+1) {
          ampi_get_idx(buf, &i, &ampi_tape->idx[count+i]);
      }
    }

    int temp=AMPI_Sendrecv_replace_f(tmp, count, datatype, dest, sendtag, source, recvtag, comm, status);

    for(i=0;i<count;i=i+1) {
        ampi_set_val(buf, &i, &tmp[i]);
    }
    free(tmp);
    return temp;
}

int AMPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status){
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
  if(recvtype!=AMPI_DOUBLE) {
    return MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf,recvcount,recvtype,source,recvtag,comm,status);
  }

    double * sendtmp = (double*) malloc(sizeof(double)*sendcount);
    double * recvtmp = (double*) malloc(sizeof(double)*recvcount);

    int i=0;

    for(i=0;i<sendcount;i=i+1) {
        ampi_get_val(sendbuf,&i,&sendtmp[i]);
    }

    if (ampi_isTapeActive()){
      /*create actual MPI entry*/
      ampi_tape_entry* ampi_tape = ampi_create_tape(sendcount+recvcount+2);
      ampi_tape->arg = (int*) malloc(sizeof(int)*6);


      for(i=0;i<sendcount;i=i+1) {
          ampi_get_idx(sendbuf, &i, &ampi_tape->idx[i]);
      }

      ampi_tape->oc = AMPI_SENDRECV;
      ampi_tape->arg[0] = sendcount;
      ampi_tape->arg[1] = dest;
      ampi_tape->arg[2] = sendtag;
      ampi_tape->arg[3] = recvcount;
      ampi_tape->arg[4] = source;
      ampi_tape->arg[5] = recvtag;
      ampi_tape->comm   = comm;

      ampi_create_dummies(recvbuf, &recvcount);
      ampi_create_tape_entry((void*)ampi_tape);

      for(i=0;i<recvcount;i=i+1) {
        ampi_get_idx(recvbuf, &i, &ampi_tape->idx[sendcount+i]);
      }
    }

    int temp=AMPI_Sendrecv_f(sendtmp, sendcount, sendtype, dest, sendtag, recvtmp,recvcount,recvtype,source,recvtag,comm,status);

    for(i=0;i<recvcount;i=i+1) {
      ampi_set_val(recvbuf, &i, &recvtmp[i]);
    }

    free(recvtmp);
    free(sendtmp);
    return temp;
}

int AMPI_Start(AMPI_Request *mpi_request_p) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif

    AMPI_GET_REQUEST(mpi_request, mpi_request_p);
    int ret = 0;
    if(NULL != mpi_request->va) {
        if (ampi_isTapeActive()){
          ampi_tape_entry* ampi_tape=0;
          ampi_tape=mpi_request->va;
          if(ampi_tape->oc==AMPI_SEND_INIT) {
              ret = AMPI_Isend(mpi_request->buf,ampi_tape->arg[0],MPI_DOUBLE,ampi_tape->arg[1],ampi_tape->tag,ampi_tape->comm,mpi_request_p);
          }
          else if(ampi_tape->oc==AMPI_RECV_INIT) {
              ret = AMPI_Irecv(mpi_request->buf,ampi_tape->arg[0],MPI_DOUBLE,ampi_tape->arg[1],ampi_tape->tag,ampi_tape->comm,mpi_request_p);
          }
          else {
              printf("Active start No opcode: %d\n",ampi_tape->oc);
          }
        }
    } else {
        ret =  MPI_Start(&mpi_request->mpiRequest);
    }

    return ret;
}

int AMPI_Startall(int count, AMPI_Request array_of_requests[]) {
#ifdef AMPI_COUNT_COMMS
    ampi_comm_count=ampi_comm_count+1;
#endif
    int i=0;
    for(i=0;i<count;i=i+1) AMPI_Start(&array_of_requests[i]);
    return 0;
}

void ampi_interpret_tape(void* handle){
    ampi_tape_entry* ampi_tape = (ampi_tape_entry*)handle;
    int j=0;
    int i=0;
    double *tmp_d;
    double *tmp_d_recv;
    double *tmp_d_send;
    MPI_Comm comm;
    MPI_Op op = MPI_SUM;
    comm = MPI_COMM_WORLD;
    MPI_Status status;
    /*ampi_vac=ampi_vac-1;*/
    /*while(ampi_tape[ampi_vac].oc == MPI_DUMMY)*/
    /*ampi_vac=ampi_vac-1;*/
    /*i=ampi_vac;*/
#ifdef AMPI_DEBUG
    printf("AMPI_TAPE Interpreter OC: %s\n", AMPI_OPCODES[ampi_tape[i].oc]);
    printf("--------------------------------START\n");
#endif
    switch(ampi_tape->oc){
  case AMPI_SEND : {
      tmp_d = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
      AMPI_Send_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->tag, comm);
      for(j=0;j<ampi_tape->arg[0];j=j+1) {
          ampi_set_adj(&ampi_tape->idx[j], &tmp_d[j]);
#ifdef AMPI_DEBUG
          printf("SEND: %ld %e\n", ampi_tape->idx[j], tmp_d[j]);
#endif
      }
      free(tmp_d);
      break;
        }
  case AMPI_RECV : {
      tmp_d = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
      for(j=0;j<ampi_tape->arg[0];j=j+1) {
          ampi_get_adj(&ampi_tape->idx[j], &tmp_d[j]);
#ifdef AMPI_DEBUG
          printf("RECV: %ld %e\n", ampi_tape->idx[j], tmp_d[j]);
#endif
      } 
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
      AMPI_Recv_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->tag, comm,&status);
      free(tmp_d);
      break;
        }
  case AMPI_BRECV : {
      tmp_d = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
      for(j=0;j<ampi_tape->arg[0];j=j+1) {
          ampi_get_adj(&ampi_tape->idx[j], &tmp_d[j]);
#ifdef AMPI_DEBUG
          printf("BRECV: %ld %e\n", ampi_tape->idx[j], tmp_d[j]);
#endif
      } 
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
      AMPI_Brecv_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->tag, comm,&status);
      free(tmp_d);
      break;
        }
  case AMPI_ISEND : {
       /*tmp_d = malloc(sizeof(double)*ampi_tape[i].arg[0]);*/
       /*if(!tape[i].request->r.aw) {*/
       tmp_d = (double*) ampi_tape->request->a;
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
       AMPI_Isend_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->tag, comm, &ampi_tape->request);
       for(j = 0 ; j < ampi_tape->arg[0] ; j++) {
           ampi_set_adj(&ampi_tape->idx[j], &tmp_d[j]);
         }
       free(tmp_d);
       break;
         }
  case AMPI_IRECV : {
       /*tmp_d = malloc(sizeof(double)*ampi_tape->arg[0]);*/
       tmp_d = (double*) ampi_tape->request->a;
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
       /*if(tape->request->r.aw) {*/

       /*}*/
       /*else {*/
           AMPI_Irecv_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->tag, comm, &ampi_tape->request);
           /*}*/
       free(tmp_d);
       break;
         }
  case AMPI_WAIT : {
      tmp_d = (double*) malloc(sizeof(double)*ampi_tape->request->va->arg[0]);
      if(ampi_tape->request->oc == AMPI_IR) {
          for(j = 0 ; j < ampi_tape->request->va->arg[0] ; j++) {
            ampi_get_adj(&ampi_tape->request->va->idx[j], &tmp_d[j]);
          }
#ifdef AMPI_DEBUG
          printf("AMPI_Wait_interpret: ");
          printf("%d ", ampi_tape[ampi_tape->arg[0]].arg[0]);
          for(j = 0 ; j < ampi_tape[ampi_tape->arg[0]].arg[0] ; j++) {
        printf("%e ", tmp_d[j]);
          }
          printf("\n");
#endif
      }
      ampi_tape->request->a = tmp_d;
      ampi_tape->request->tag=ampi_tape->request->va->tag;
#ifndef NO_COMM_WORLD 
      ampi_tape->request->comm=MPI_COMM_WORLD;
#endif
      AMPI_Wait_b(&ampi_tape->request,&status);
      ampi_tape->request->va->request = ampi_tape->request;
      break;
        }
  case AMPI_BCAST : {
           int rank=0;
           int root=ampi_tape->arg[1];
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
           MPI_Comm_rank(comm,&rank);
           
      tmp_d = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
      if(rank!=root) {
        for(j=0;j<ampi_tape->arg[0];j=j+1) {
          ampi_get_adj(&ampi_tape->idx[j], &tmp_d[j]);
        } 
      }
      else {
        for(j=0;j<ampi_tape->arg[0];j=j+1) tmp_d[j]=0;
      }
       AMPI_Bcast_b(tmp_d, ampi_tape->arg[0], MPI_DOUBLE, root, comm);
      if(rank==root) {
         for(j=0 ; j<ampi_tape->arg[0] ; j++) {
           ampi_set_adj(&ampi_tape->idx[j],&tmp_d[j]);
         }
      }
       free(tmp_d);
       break;
         }
  case AMPI_REDUCE : {
        int rank=0;
        int root=ampi_tape->arg[1];
        MPI_Comm_rank(comm,&rank);
        if(ampi_tape->arg[2] == AMPI_REDUCE_ADD)
            op = MPI_SUM;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MUL)
            op = MPI_PROD;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MIN)
            op = MPI_MIN;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MAX)
            op = MPI_MAX;
        tmp_d_send = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
        tmp_d_recv = (double*) malloc(sizeof(double)*ampi_tape->arg[0]); /* needs to be allocated for reverse function */
        if(root == rank) {
          for(j=0;j<ampi_tape->arg[0];j=j+1) {
            ampi_get_adj(&ampi_tape->idx[ampi_tape->arg[0] + j], &tmp_d_recv[j]);
          }
        }

        for(j=0;j<ampi_tape->arg[0];j=j+1) {
          tmp_d_send[j]=0;
        }

#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
        AMPI_Reduce_b(tmp_d_send, tmp_d_recv, ampi_tape->arg[0], MPI_DOUBLE, op, ampi_tape->arg[1], comm, ampi_tape->stack);
        for(j=0 ; j<ampi_tape->arg[0] ; j++) {
            ampi_set_adj(&ampi_tape->idx[j],&tmp_d_send[j]);
        }
        free(tmp_d_send);
        free(tmp_d_recv);
        break;
          }
  case AMPI_ALLREDUCE : {
        if(ampi_tape->arg[2] == AMPI_REDUCE_ADD)
            op = MPI_SUM;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MUL)
            op = MPI_PROD;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MIN)
            op = MPI_MIN;
        if(ampi_tape->arg[2] == AMPI_REDUCE_MAX)
            op = MPI_MAX;
        tmp_d_send = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
        tmp_d_recv = (double*) malloc(sizeof(double)*ampi_tape->arg[0]);
        for(j=0;j<ampi_tape->arg[0];j=j+1)
            ampi_get_adj(&ampi_tape->idx[ampi_tape->arg[0] + j], &tmp_d_recv[j]);
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif

#ifdef AMPI_DEBUG
        printf("AMPI_Allreduce tmp_d_recv: ");
        for(j=0 ; j<ampi_tape->arg[0] ; j++) {
            /*if(tmp_d_recv[j]!=tmp_d_recv[j]) tmp_d_recv[j]=0;*/
            printf("%e ",tmp_d_recv[j]);
        }
        printf("\n");
#endif
        AMPI_Allreduce_b(tmp_d_send, tmp_d_recv, ampi_tape->arg[0], MPI_DOUBLE, op, comm, ampi_tape->stack);
#ifdef AMPI_DEBUG
        printf("AMPI_Allreduce tmp_d_send: ");
        for(j=0 ; j<ampi_tape->arg[0] ; j++) {
            printf("%e ",tmp_d_send[j]);
        }
        printf("\n");
#endif
        for(j=0 ; j<ampi_tape->arg[0] ; j++) {
            ampi_set_adj(&ampi_tape->idx[j],&tmp_d_send[j]);
        }
        free(tmp_d_send);
        free(tmp_d_recv);
        break;
          }
  case AMPI_SCATTER : {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       int rank=0;
       int root=ampi_tape->arg[0];
       int crecv=ampi_tape->arg[1];
       int csend=ampi_tape->arg[2];
       MPI_Comm_size(comm,&size);
       MPI_Comm_rank(comm,&rank);
       double *sendbuf=NULL;
       int sendSize=0;
       if(rank == root) {
         sendSize = csend*size;
         sendbuf = (double*) malloc(sizeof(double)*sendSize);
       }
       double *recvbuf=(double*) malloc(sizeof(double)*crecv);

       for(j=0;j<crecv;j++) {
         ampi_get_adj(&ampi_tape->idx[sendSize+j],&recvbuf[j]);
       }
       AMPI_Scatter_b(sendbuf,csend,MPI_DOUBLE,recvbuf,crecv,MPI_DOUBLE,root,comm);

       if(rank == root) {
         for(j=0;j<sendSize;j++) {
           ampi_set_adj(&ampi_tape->idx[j],&sendbuf[j]);
         }
       }
       if(rank == root) {
         free(sendbuf);
       }
       free(recvbuf);
       break;
           }
  case AMPI_SCATTERV : {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       int rank=0;
       MPI_Comm_size(comm,&size);
       MPI_Comm_rank(comm,&rank);
       int root=ampi_tape->arg[0];
       int crecv=ampi_tape->arg[1];
       int ctotal=ampi_tape->arg[2];
       int* csends=&ampi_tape->arg[3];
       int* cdispl=&ampi_tape->arg[3 + size];
       double *sendbuf=NULL;
       int sendSize=0;
       if(rank == root) {
         sendSize = ctotal;
         sendbuf = (double*) malloc(sizeof(double)*ctotal);
       }
       double *recvbuf=(double*) malloc(sizeof(double)*crecv);

       for(j=0;j<crecv;j++) {
         ampi_get_adj(&ampi_tape->idx[sendSize+j],&recvbuf[j]);
       }
       MPI_Gatherv(recvbuf, crecv,MPI_DOUBLE, sendbuf, csends, cdispl,MPI_DOUBLE,root,comm);

       if(rank == root) {
         for(j=0;j<sendSize;j++) {
           ampi_set_adj(&ampi_tape->idx[j],&sendbuf[j]);
         }
       }
       if(rank == root) {
         free(sendbuf);
       }
       free(recvbuf);
       break;
    }
  case AMPI_GATHER: {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       int rank=0;
       int root=ampi_tape->arg[0];
       int crecv=ampi_tape->arg[1];
       int csend=ampi_tape->arg[2];
       MPI_Comm_size(comm,&size);
       MPI_Comm_rank(comm,&rank);
       double *sendbuf=(double*)malloc(sizeof(double)*csend);
       double *recvbuf=NULL;
       if(rank == root) {
         recvbuf = (double*)malloc(sizeof(double)*crecv*size);
         for(j=0;j<crecv*size;j++) {
           ampi_get_adj(&ampi_tape->idx[csend + j],&recvbuf[j]);
         }
       }

       AMPI_Gather_b(sendbuf,csend,MPI_DOUBLE,recvbuf,crecv,MPI_DOUBLE,root,comm);
       for(j=0;j<csend;j++) {
         ampi_set_adj(&ampi_tape->idx[j],&sendbuf[j]);
       }
       free(sendbuf);
       if(rank == root) {
         free(recvbuf);
       }
         break;
           }
  case AMPI_GATHERV : {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       int rank=0;
       MPI_Comm_size(comm,&size);
       MPI_Comm_rank(comm,&rank);
       int root=ampi_tape->arg[0];
       int csend=ampi_tape->arg[1];
       int  ctotal=ampi_tape->arg[2];
       int* crecvs=&ampi_tape->arg[3];
       int* cdispl=&ampi_tape->arg[3 + size];
       double *sendbuf=(double*)malloc(sizeof(double)*csend);
       double *recvbuf=NULL;
       if(rank == root) {
         recvbuf = (double*)malloc(sizeof(double)*ctotal);
         for(j=0;j<ctotal;j++) {
           ampi_get_adj(&ampi_tape->idx[csend + j],&recvbuf[j]);
         }
       }

       MPI_Scatterv(recvbuf, crecvs, cdispl, MPI_DOUBLE, sendbuf, csend, MPI_DOUBLE, root, comm);
       for(j=0;j<csend;j++) {
         ampi_set_adj(&ampi_tape->idx[j],&sendbuf[j]);
       }
       free(sendbuf);
       if(rank == root) {
         free(recvbuf);
       }
       break;
    }
  case AMPI_ALLGATHER: {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       int crecv=ampi_tape->arg[0];
       int csend=ampi_tape->arg[1];
       MPI_Comm_size(comm,&size);
       double *sendbuf=(double*)malloc(sizeof(double)*csend*size);
       double *recvbuf=(double*)malloc(sizeof(double)*crecv*size);
       for(j=0;j<crecv*size;j++) {
         ampi_get_adj(&ampi_tape->idx[csend + j],&recvbuf[j]);
       }

       MPI_Alltoall(recvbuf,crecv,MPI_DOUBLE,sendbuf,csend,MPI_DOUBLE,comm);
       for(j=0;j<csend;j++) {
         double adj = 0.0;
         for(i=0;i<size;i++) {
           adj = adj + sendbuf[j + i * csend];
         }
         ampi_set_adj(&ampi_tape->idx[j],&adj);
       }
       free(sendbuf);
       free(recvbuf);
       break;
    }
  case AMPI_ALLGATHERV : {
#ifdef NO_COMM_WORLD
       comm=ampi_tape->comm;
#endif
       int size=0;
       MPI_Comm_size(comm,&size);
       int  ctotal=ampi_tape->arg[0];
       int* crecvs=&ampi_tape->arg[1];
       int* crdisp=&ampi_tape->arg[1 + size];
       int* csends=&ampi_tape->arg[1 + 2 * size];/* uniform sizes in the send structure */
       int* csdisp=&ampi_tape->arg[1 + 3 * size];

       double *sendbuf=(double*)malloc(sizeof(double)*csends[0]*size);
       double *recvbuf=(double*)malloc(sizeof(double)*ctotal);
       for(j=0;j<ctotal;j++) {
         ampi_get_adj(&ampi_tape->idx[csends[0] + j],&recvbuf[j]);
       }

       MPI_Alltoallv(recvbuf, crecvs, crdisp, MPI_DOUBLE, sendbuf, csends, csdisp, MPI_DOUBLE, comm);
       for(j=0;j<csends[0];j++) {
         double adj = 0.0;
         for(i=0;i<size;i++) {
           adj = adj + sendbuf[j + i * csends[0]];
         }
         ampi_set_adj(&ampi_tape->idx[j],&adj);
       }
       free(sendbuf);
       free(recvbuf);
       break;
    }
  case AMPI_SENDRECVREPLACE : {
    int count = ampi_tape->arg[0];
    tmp_d = (double*) malloc(sizeof(double)* count);
      /*take adjoints out of the tape in the send buffer*/
      for(j=0;j<count;j=j+1) {
          ampi_get_adj(&ampi_tape->idx[count + j], &tmp_d[j]);
#ifdef AMPI_DEBUG
          printf("SENDRECV_B: %ld %e\n", ampi_tape->idx[count + j], tmp_d[j]);
#endif
      } 
#ifdef NO_COMM_WORLD
      comm=ampi_tape->comm;
#endif
      AMPI_Sendrecv_replace_b(tmp_d, count, MPI_DOUBLE, ampi_tape->arg[2], ampi_tape->arg[3], ampi_tape->arg[1], ampi_tape->tag, comm, &status);

                  /*two entries for sendrecvreplace, decrease tape index*/
      for(j=0;j<count;j=j+1) {
          ampi_set_adj(&ampi_tape->idx[j], &tmp_d[j]);
#ifdef AMPI_DEBUG
          printf("SEND: %ld %e\n", ampi_tape->idx[j], tmp_d[j]);
#endif
      }
      free(tmp_d);
      break;
       }
      case AMPI_SENDRECV: {
          int recvcount = ampi_tape->arg[3];
          int sendcount = ampi_tape->arg[0];
          double *sendbuf = (double*) malloc(sizeof(double)*sendcount);
          double *recvbuf = (double*) malloc(sizeof(double)*recvcount);
          for (j=0; j < recvcount; j=j+1){
              ampi_get_adj(&ampi_tape->idx[sendcount+j],&recvbuf[j]);
            }
          comm = ampi_tape->comm;
          AMPI_Sendrecv_b(sendbuf, sendcount, MPI_DOUBLE, ampi_tape->arg[1], ampi_tape->arg[2], recvbuf, recvcount, MPI_DOUBLE, ampi_tape->arg[4], ampi_tape->arg[5],comm,&status);

          for (j=0;j<sendcount;j=j+1){
              ampi_set_adj(&ampi_tape->idx[j], &sendbuf[j]);
            }
          free(sendbuf);
          free(recvbuf);
          break;
        }
  case AMPI_SEND_INIT : {
      break;
       }
  case AMPI_RECV_INIT : {
      break;
       }
  default: {
         printf("Warning: Missing opcode in the AMPI tape interpreter for %d.\n", ampi_tape->oc);
         break;
          }

    }
}
void ampi_reset_entry(void* handle){
  ampi_release_tape((ampi_tape_entry*)handle);
}

void ampi_release_tape(ampi_tape_entry* ampi_tape) {
  if(ampi_tape->arg != NULL) {
    free(ampi_tape->arg);
    ampi_tape->arg = NULL;
  }
  if(ampi_tape->idx != NULL) {
    free(ampi_tape->idx);
    ampi_tape->idx = NULL;
  }
  if(ampi_tape->stack != NULL) {
    AMPI_stack_delete(ampi_tape->stack);
    ampi_tape->stack = NULL;
  }
  if(ampi_tape->oc == AMPI_WAIT) {
    /* wait does not need to delete its request */
    ampi_tape->request = NULL;
  } else {
    if(ampi_tape->request != NULL) {
      free(ampi_tape->request);
      ampi_tape->request = NULL;
    }
  }

  free(ampi_tape);
}

ampi_tape_entry* ampi_create_tape(long int size) {
  ampi_tape_entry* ampi_tape = (ampi_tape_entry*) calloc(1, sizeof(ampi_tape_entry));
  ampi_tape->arg = NULL;
  ampi_tape->idx = NULL;
  ampi_tape->stack = NULL;
  ampi_tape->request = NULL;
  if(0 != size) {
    ampi_tape->idx = (INT64*) malloc(sizeof(INT64) * size);
  }

  return ampi_tape;
}
