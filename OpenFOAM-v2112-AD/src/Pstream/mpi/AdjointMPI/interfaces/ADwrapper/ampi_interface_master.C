#if defined(DAOF_AD_MODE_A1S)

#include "ADdefines.H"
#include "ampi_tape.hpp"

#define INT64 long int

typedef Foam::ADmode AD_MODE;
typedef Foam::ADtype AD_TYPE;

//#ifndef DCO_AMPI
//void ampi_interpret_tape(long int idx) {}
//#endif
//long int ampi_counter=0;

void ampi_get_val(void *buf, int *i, double *x) {
    *x = AD::value((static_cast<AD_TYPE*>(buf))[*i]);
}
void ampi_set_val(void* buf, int *i, double *v) {
    AD::value((static_cast<AD_TYPE*>(buf))[*i]) = *v;
}

void ampi_get_idx(void *buf, int *i, INT64 *idx) {
    *idx = AD::tapeIndex((static_cast<AD_TYPE*>(buf))[*i]);
}

void ampi_get_adj(INT64 *idx, double *x) {
    if(*idx!=0) *x = AD::adjointFromIndex(*idx);
}
void ampi_set_adj(INT64 *idx, double *x) {
    if(*idx!=0) AD::adjointFromIndex(*idx) += *x;
}


/*extern "C" */void ampi_reset_entry(long int handle);

/*struct AMPI_data : AD_MODE::callback_object_t::callback_object_base  {
  void* handle;

  AMPI_data(){}
  AMPI_data(const AMPI_data *other) {
    handle=other->handle;
  }
  virtual ~AMPI_data() {
    ampi_reset_entry(handle);
  }
};

void ampi_tape_wrapper(AMPI_data *data) {
  ampi_interpret_tape(data->handle);
}*/


void ampi_create_tape_entry(void* handle) {
    if(AD::isTapeActive()) {
        AD::insertAdjointCallback(
            [handle](){ ampi_interpret_tape(handle); }, /* interpret */
            [handle](){ /*ampi_reset_entry(handle);*/ } /* destroy   */
        );
    }
}

/*void ampi_create_dummies(void *buf, int *size) {
    AD_TYPE *values=static_cast<AD_TYPE*>(buf);

    for(int i=0; i<*size; ++i) {
        AD_TYPE &dummy=values[i];
        dummy=0;
        AD::registerInputVariable(dummy);
    }
}*/

void ampi_create_dummies_displ(void *buf, int* displ, int *size) {
    if (AD::isTapeActive()){

        AD_TYPE *values=static_cast<AD_TYPE*>(buf);

        for(int i=0;i<*size;++i) {
            AD_TYPE& dummy=values[*displ + i];
            dummy=0;
            AD::registerInputVariable(dummy);
        }
    }
}

int ampi_isTapeActive () {
    return AD::isTapeActive();
}

#endif
