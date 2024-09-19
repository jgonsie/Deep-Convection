#ifndef ADdefines_dco_H
#define ADdefines_dco_H

#include "label.H"
//#ifndef DCO_AUTO_SUPPORT
#define DCO_AUTO_SUPPORT
//#endif
//#ifndef DCO_DISABLE_AVX2_WARNING
#define DCO_DISABLE_AVX2_WARNING
//#endif

#define DCO_TAPE_USE_LONG_INT
#define DCO_SQRT_AVOID_DIVZERO

#if defined(DAOF_AD_MODE_A1S)
    #define DAOF_SDLS
    #define DCO_AMPI
    #define AMPI_COUNT
#elif defined(DAOF_AD_MODE_T1V)
    #define DCO_T1V_SIZE 16
#elif defined(DAOF_AD_MODE_T2A1S)
    #define DAOF_SDLS
#endif

#include <dco.hpp>
#define AD_TOOL_VERSION_SHORT dco_cpp DCO_VERSION_SHORT

#include <memory>

#define DAOF_MODE_ACTIVE

namespace Foam
{
#if defined(DAOF_AD_MODE_A1S)
    using ADmode = dco::ga1s<double>;
#elif defined(DAOF_AD_MODE_T1S)
    using ADmode = dco::gt1s<double>;
#elif defined(DAOF_AD_MODE_T1V)
    using ADmode = dco::gt1v<double,DCO_T1V_SIZE>;
#elif defined(DAOF_AD_MODE_T2A1S)
    using ADmode = dco::ga1s<dco::gt1s<double>::type >;
#elif defined(DAOF_AD_MODE_T2T1S)
    using ADmode = dco::gt1s<dco::gt1s<double>::type >;
#else
    using ADmode = dco::mode<double>;
    #undef DAOF_MODE_ACTIVE
#endif // DAOF_AD_MODE
    using ADtype = ADmode::type;
    using doubleScalar = ADtype;
    using doubleScalarPassive = double;
}

// ADJOINT INTERFACE
//namespace dco_foam = dco; // offer dco_foam namespace for Scalar.H

namespace AD{
namespace operations = dco::internal;

template<typename = void>
inline const Foam::ADmode::value_t& value(const Foam::doubleScalar& x){
    return dco::value(x);
}

template<typename = void>
inline Foam::ADmode::value_t& value(Foam::doubleScalar& x){
    return dco::value(x);
}

template<typename = void>
inline const Foam::ADmode::passive_t& passiveValue(const Foam::doubleScalar& x){
    return dco::passive_value(x);
}

template<typename = void>
inline Foam::ADmode::passive_t& passiveValue(Foam::doubleScalar& x){
    return dco::passive_value(x);
}

#ifdef DAOF_MODE_ACTIVE
template<typename = void>
inline Foam::ADmode::derivative_t& derivative(const Foam::doubleScalar& x){
    return dco::derivative(x);
}

template<typename = void>
inline Foam::ADmode::derivative_t& derivative(Foam::doubleScalar& x){
    return dco::derivative(x);
}

#ifdef DAOF_AD_MODE_A1S
typedef Foam::ADmode::tape_t::position_t position_t;

template<typename = void>
inline Foam::ADmode::derivative_t& adjointFromIndex(Foam::label idx){
    return Foam::ADmode::global_tape->_adjoint(idx);
}

template<typename = void>
inline Foam::label tapeIndex(const Foam::doubleScalar& x){
    return dco::tape_index(x);
}

template<typename = void>
inline void registerInputVariable(const Foam::doubleScalar& x){
    Foam::ADmode::global_tape->register_variable(const_cast<Foam::doubleScalar&>(x));
}

template<class ITER_T>
inline void registerInputVariable(const ITER_T& begin, const ITER_T& end){
    for(ITER_T t = begin; t != end; t++){
        AD::registerInputVariable(*t);
    }
}

template<typename = void>
inline void registerOutputVariable(Foam::doubleScalar& x){
    Foam::ADmode::global_tape->register_output_variable(x);
}

template<class ITER_T>
inline void registerOutputVariable(const ITER_T& begin, const ITER_T& end){
    for(ITER_T t = begin; t != end; t++){
        AD::registerOutputVariable(*t);
    }
}

template<typename = void>
inline Foam::ADmode::tape_t& createGlobalTape(Foam::label tapeSizeMB=1024, double split=0.75){
#ifdef DAOF_AD_TOOL_DCO_CPP_DEV
    dco::tape_options opts;
    opts.set_blob_size_in_mbyte(tapeSizeMB);
    Foam::ADmode::global_tape = Foam::ADmode::tape_t::create(opts); // ignore split
#else
    Foam::label entries = 1024l*1024*tapeSizeMB/sizeof(Foam::doubleScalar);
    Foam::ADmode::global_tape = Foam::ADmode::tape_t::create(Foam::label(split*entries),Foam::label((1-split)*entries));
#endif
    return *Foam::ADmode::global_tape;
}

template<typename = void>
inline position_t getTapePosition(){
    return Foam::ADmode::global_tape->get_position();
}

template<typename = void>
inline void interpretTape(){
    Foam::ADmode::global_tape->interpret_adjoint();
}

template<typename = void>
inline void interpretTapeTo(const position_t& to){
    Foam::ADmode::global_tape->interpret_adjoint_to(to);
}

template<typename = void>
inline void interpretTapeFromTo(const position_t& from, const position_t& to){
    Foam::ADmode::global_tape->interpret_adjoint_from_to(from,to);
}

template<typename = void>
inline void resetTape(){
    Foam::ADmode::global_tape->reset();
}

template<typename = void>
inline void resetTapeTo(const position_t& to){
    Foam::ADmode::global_tape->reset_to(to);
}

template<typename = void>
inline void zeroAdjointVector(){
    Foam::ADmode::global_tape->zero_adjoints();
}

template<typename = void>
inline void zeroAdjointVectorTo(const position_t& to){
    Foam::ADmode::global_tape->zero_adjoints_to(to);
}

template<typename = void>
inline void zeroAdjointVectorFromTo(const position_t& from, const position_t& to){
    Foam::ADmode::global_tape->zero_adjoints_from_to(from,to);
}

template<typename = void>
inline bool isTapeActive(){
    return Foam::ADmode::global_tape != NULL && Foam::ADmode::global_tape->is_active();
}

template<typename = void>
inline bool switchTapeToPassive(){
    bool was_active = AD::isTapeActive();
    if(Foam::ADmode::global_tape != NULL){
        Foam::ADmode::global_tape->switch_to_passive();
    }
    return was_active;
}

template<typename = void>
inline bool switchTapeToActive(){
    bool was_active = AD::isTapeActive();
    if(Foam::ADmode::global_tape != NULL){
        Foam::ADmode::global_tape->switch_to_active();
    }
    return was_active;
}

template<typename = void>
inline bool switchTapeTo(bool on){
    if(on){
        return AD::switchTapeToActive();
    }else{
        return AD::switchTapeToPassive();
    }
}

template<class T>
void passiveFunctionWrapper(T&& fun){
    bool wasActive = switchTapeToPassive();
    fun();
    switchTapeTo(wasActive);
}

template<typename = void>
inline void removeTape(){
    if(Foam::ADmode::global_tape){
        Foam::ADmode::tape_t::remove( Foam::ADmode::global_tape );
    }
}


namespace internal
{
    template<class T, class D>
    struct ExternalFunctionDataObject : public Foam::ADmode::callback_object_t{
        std::shared_ptr<T> fun;
        std::shared_ptr<D> dest;
        ~ExternalFunctionDataObject(){
            (*dest.get())();
        }
    };

    template<class T, class D>
    void reverseFunctionWrapper(ExternalFunctionDataObject<T,D> *d) {
        (*d->fun.get())();
    }
}

static constexpr auto dDefault = [](){};
template<class T, class D = decltype(dDefault) const&>
void insertAdjointCallback(T&& fun, D&& dest = dDefault)
{
    using TT = std::remove_reference_t<T>;
    using DD = std::remove_reference_t<D>;
    auto* dataObjectPtr = Foam::ADmode::global_tape->create_callback_object<AD::internal::ExternalFunctionDataObject<TT,DD>>();
    dataObjectPtr->fun = std::make_shared<TT>(std::forward<T>(fun));
    dataObjectPtr->dest = std::make_shared<DD>(std::forward<D>(dest));
    Foam::ADmode::global_tape->insert_callback(AD::internal::reverseFunctionWrapper<TT,DD>,dataObjectPtr);
}
#endif
#endif
}

#define AD_ENABLE_TYPE_CONSTRUCTION_FROM DCO_ENABLE_TYPE_CONSTRUCTION_FROM

namespace dco{
    namespace internal{
        template<typename = void>
        inline Foam::doubleScalar cbrt(const Foam::doubleScalar& s){
            using std::pow;
            return pow(s,Foam::doubleScalar(1.0/3.0));
        }
        template<typename = void>
        inline Foam::doubleScalar lgamma(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::lgamma(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar tgamma(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::tgamma(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar j0(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::j0(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar j1(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::j1(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar y0(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::y0(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar y1(const Foam::doubleScalar& s){
            return static_cast<Foam::doubleScalar>(::y1(AD::passiveValue(s)));
        }
        template<typename = void>
        inline Foam::doubleScalar fmod(const Foam::doubleScalar& x, const Foam::doubleScalar& y){
            // Towara: hacky way to implement active fmod
            return static_cast<Foam::doubleScalar>(x - Foam::label(AD::passiveValue(x/y))*y);
        }
    }
}
#endif
