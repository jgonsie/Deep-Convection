#if defined(DAOF_AD_MODE_A1S)

#include "ADdefines.H"
#include "ampi_tape.hpp"

#define INT64 long int

typedef Foam::ADmode AD_MODE;
typedef AD_MODE::type AD_TYPE;

#ifndef DCO_AMPI
void ampi_interpret_tape(long int idx) {}
#endif
//long int ampi_counter=0;

void ampi_get_val(void *buf, int *i, double *x) {
    *x=static_cast<AD_TYPE*>(buf)[*i]._value();
}
void ampi_set_val(void* buf, int *i, double *v) {
    AD_TYPE &dummy= static_cast<AD_TYPE*>(buf)[*i];
    *const_cast<double*>(&(dummy._value())) = *v;
}

void ampi_get_idx(void *buf, int *i, INT64 *idx) {
    AD_TYPE &var = static_cast<AD_TYPE*>(buf)[*i];
    *idx = dco::tapeIndex(var);
}

void ampi_get_adj(INT64 *idx, double *x) {
    if(*idx!=0) *x = AD_MODE::global_tape->_adjoint(*idx);
}
void ampi_set_adj(INT64 *idx, double *x) {
    if(*idx!=0) AD_MODE::global_tape->_adjoint(*idx) += *x;
}


/*extern "C" */void ampi_reset_entry(long int handle);

struct AMPI_data : AD_MODE::callback_object_t::callback_object_base  {
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
}


void ampi_create_tape_entry(void* handle) {
    if(AD_MODE::global_tape == NULL || !AD_MODE::global_tape->is_active()) {
        return;
    }
    AMPI_data *tmp=new AMPI_data();
    tmp->handle=handle;
    AD_MODE::global_tape->insert_callback(
        &ampi_tape_wrapper,
        AD_MODE::global_tape->create_callback_object<AMPI_data>(tmp)
    );
}

void ampi_create_dummies(void *buf, int *size) {
    AD_TYPE *values=static_cast<AD_TYPE*>(buf);

    for(int i=0; i<*size; ++i) {
        AD_TYPE &dummy=values[i];
        dummy=0;
        AD_MODE::global_tape->register_variable(dummy);
    }
}

void ampi_create_dummies_displ(void *buf, int* displ, int *size) {
  if (NULL != AD_MODE::global_tape && AD_MODE::global_tape->is_active()){

    AD_TYPE *values=static_cast<AD_TYPE*>(buf);

    for(int i=0;i<*size;++i) {
      AD_TYPE& dummy=values[*displ + i];
      dummy=0;
      AD_MODE::global_tape->register_variable(dummy);
    }
  }
}

int ampi_isTapeActive () {
    if (NULL != AD_MODE::global_tape) {
#ifdef DCO_ALLOW_TAPE_SWITCH_OFF
        return AD_MODE::global_tape->is_active();
#else
        return 1;
#endif
    } else {
        return 0;
    }
}

#endif
