#ifndef CHECKBASEOBJECT_H
#define CHECKBASEOBJECT_H

//#include "CheckObject.H"
//#include "CheckObjectScalar.H"
#include "string.H"
#include "scalar.H"

class CheckBaseObject
{
  public:
#if defined(DAOF_AD_MODE_A1S)
    virtual void storeAdjoints() = 0;
    virtual double calcNormOfStoredAdjoints() = 0;
    virtual void restoreAdjoints() = 0;
    virtual void registerAdjoints() = 0;
    virtual void registerAsOutput() = 0;
    // virtual void storeTapeIndex()=0;
#endif
    virtual Foam::string name() const { return "Dummy string"; }
    virtual void addCheckpoint() = 0;
    virtual void replaceCheckpoint(int i) = 0;
    virtual void restoreCheckpoint(int i) = 0;
    virtual double getObjectSize() = 0;
    virtual ~CheckBaseObject() {}
  protected:
    typedef std::decay_t<decltype(AD::value(std::declval<Foam::scalar>()))> AD_BASE_TYPE;
};

#endif
