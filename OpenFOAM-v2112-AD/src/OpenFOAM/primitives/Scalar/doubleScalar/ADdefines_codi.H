#ifndef ADdefines_codi_H
#define ADdefines_codi_H

#include "label.H"
#include <codi.hpp>
#define AD_TOOL_VERSION_SHORT CoDiPack 1.9.2
#include <memory>

#define DAOF_MODE_ACTIVE
namespace Foam
{
#if defined(DAOF_AD_MODE_A1S)
    using ADmode = codi::ActiveReal<codi::JacobiTape<codi::JacobiTapeTypes<codi::ReverseTapeTypes<double, double, codi::LinearIndexHandler<label> >, codi::SingleChunkVector>>>; // RealReverse but with long tape index
#elif defined(DAOF_AD_MODE_T1S)
    using ADmode = codi::RealForward;
#else
    using ADmode = double; // passive mode
    #undef DAOF_MODE_ACTIVE
#endif // DAOF_AD_MODE
    using ADtype = ADmode;
    using doubleScalar = ADtype;
    using doubleScalarPassive = double;
}

#if defined(DAOF_AD_MODE_A1S)
    #define DAOF_SDLS
    #define AMPI_COUNT
#endif

namespace AD{
  namespace operations = codi;

template<typename = void>
inline const Foam::doubleScalar::Real& value(const Foam::doubleScalar& x){
    return x.value();
}

template<typename = void>
inline Foam::doubleScalar::Real& value(Foam::doubleScalar& x){
    return x.value();
}

template<typename = void>
inline Foam::doubleScalar::PassiveReal passiveValue(const Foam::doubleScalar& x){
    return codi::TypeTraits<Foam::doubleScalar>::getBaseValue(x);
}

#ifdef DAOF_MODE_ACTIVE

template<typename = void>
inline Foam::doubleScalar::GradientValue& derivative(const Foam::doubleScalar& x){
    return const_cast<Foam::doubleScalar::GradientValue&>(x.gradient());
}

template<typename = void>
inline Foam::doubleScalar::GradientValue& derivative(Foam::doubleScalar& x){
    return x.gradient();
}

#ifdef DAOF_AD_MODE_A1S
typedef Foam::doubleScalar::TapeType::Position position_t;

template<typename = void>
inline Foam::doubleScalar::TapeType& createGlobalTape(Foam::label tapeSizeMB=1024, double split=0.75){
    Foam::label entries = 1024l*1024*tapeSizeMB/sizeof(Foam::doubleScalar);
    Foam::doubleScalar::getGlobalTape().resize(Foam::label(split*entries),Foam::label((1-0.75)*entries));
    Foam::doubleScalar::TapeType& tape = Foam::doubleScalar::getGlobalTape();
    tape.setActive();
    return tape;
}

template<typename = void>
inline Foam::doubleScalarPassive& adjointFromIndex(Foam::label idx){
    return Foam::doubleScalar::getGlobalTape().gradient(idx);
}

template<typename = void>
inline Foam::label tapeIndex(const Foam::doubleScalar& x){
    return x.getGradientData();
}

template<typename = void>
inline void registerInputVariable(const Foam::doubleScalar& x){
    Foam::doubleScalar::getGlobalTape().registerInput(const_cast<Foam::doubleScalar&>(x));
}

template<class ITER_T>
inline void registerInputVariable(const ITER_T& begin, const ITER_T& end){
    for(ITER_T t = begin; t != end; t++){
        AD::registerInputVariable(*t);
    }
}

template<typename = void>
inline void registerOutputVariable(Foam::doubleScalar& x){
    Foam::doubleScalar::getGlobalTape().registerOutput(x);
}

template<class ITER_T>
inline void registerOutputVariable(const ITER_T& begin, const ITER_T& end){
    for(ITER_T t = begin; t != end; t++){
        AD::registerOutputVariable(*t);
    }
}

template<typename = void>
inline position_t getTapePosition(){
    return Foam::doubleScalar::getGlobalTape().getPosition();
}

template<typename = void>
inline void interpretTape(){
    Foam::doubleScalar::getGlobalTape().evaluate();
}

template<typename = void>
inline void interpretTapeTo(const position_t& to){
    Foam::doubleScalar::getGlobalTape().evaluate(Foam::doubleScalar::getGlobalTape().getPosition(),to);
}

template<typename = void>
inline void interpretTapeFromTo(const position_t& start, const position_t& end){
    Foam::doubleScalar::getGlobalTape().evaluate(start,end);
}

template<typename = void>
inline void resetTape(){
    Foam::doubleScalar::getGlobalTape().reset();
}

template<typename = void>
inline void resetTapeTo(const position_t& to){
    Foam::doubleScalar::getGlobalTape().reset(to);
}

template<typename = void>
inline void zeroAdjointVector(){
    Foam::doubleScalar::getGlobalTape().clearAdjoints();
}

template<typename = void>
inline void zeroAdjointVectorTo(const position_t& to){
    Foam::doubleScalar::getGlobalTape().clearAdjoints(Foam::doubleScalar::getGlobalTape().getPosition(),to);
}

template<typename = void>
inline void zeroAdjointVectorFromTo(const position_t& from, const position_t& to){
    Foam::doubleScalar::getGlobalTape().clearAdjoints(from,to);
}

template<typename = void>
inline bool isTapeActive(){
    return Foam::doubleScalar::getGlobalTape().isActive();
}

template<typename = void>
inline bool switchTapeToPassive(){
    bool was_active = AD::isTapeActive();
    Foam::doubleScalar::getGlobalTape().setPassive();
    return was_active;
}

template<typename = void>
inline bool switchTapeToActive(){
    bool was_active = AD::isTapeActive();
    Foam::doubleScalar::getGlobalTape().setActive();
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
    // pass
}

namespace internal
{
    template<class TT, class DD>
    struct CodiUserData{
        TT fun;
        DD dest;

        template<class T, class D>
        CodiUserData(T&& fun, D&& dest) : fun(std::forward<T>(fun)),
                                          dest(std::forward<D>(dest))
        {}
        ~CodiUserData(){
            dest();
        }
    };
    template<class TT, class DD>
    void reverse_fun_wrapper(const Foam::doubleScalar::Real* x, Foam::doubleScalar::Real* x_b, size_t m, const Foam::doubleScalar::Real* y, const Foam::doubleScalar::Real* y_b, size_t n, codi::DataStore* D)
    {
        D->getData<std::shared_ptr<CodiUserData<TT,DD>>>()->fun();
    }
}

static constexpr auto dDefault = [](){};
template<class T, class D = decltype(dDefault) const&>
void insertAdjointCallback(T&& fun, D&& dest = dDefault){
    using TT = std::remove_reference_t<T>;
    using DD = std::remove_reference_t<D>;
    codi::ExternalFunctionHelper<Foam::doubleScalar> efh;
    // make shared so destructor is only called when tape is reset
    auto cudPtr = std::make_shared<AD::internal::CodiUserData<TT,DD>>(std::forward<T>(fun),std::forward<D>(dest));
    efh.addUserData(cudPtr);
    efh.addToTape(AD::internal::reverse_fun_wrapper<TT,DD>);
}
#endif
#endif
}

#define AD_ENABLE_TYPE_CONSTRUCTION_FROM CODI_ENABLE_TYPE_CONSTRUCTOR

namespace codi{
    template<typename = void>
    inline Foam::doubleScalar log1p(const Foam::doubleScalar& s){
        return log(1.0+s);
    }
    template<typename = void>
    inline Foam::doubleScalar log10(const Foam::doubleScalar& s){
        return log(s)/std::log(10);
    }
    template<typename = void>
    inline Foam::doubleScalar asinh(const Foam::doubleScalar& s){
        return log(s+sqrt(pow(s,2)+1));
    }
    template<typename = void>
    inline Foam::doubleScalar acosh(const Foam::doubleScalar& s){
        return log(s+sqrt(pow(s,2)+1));
    }
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


#endif

