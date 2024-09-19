#ifndef CHECKOBJECTSCALAR_H
#define CHECKOBJECTSCALAR_H
#include "CheckBaseObject.H"

class CheckObjectScalar : public CheckBaseObject
{
  public:
    // specializations for scalar and vector
    CheckObjectScalar(Foam::scalar& object, bool reverseAccumulation = false);
#if defined(DAOF_AD_MODE_A1S)
    void storeAdjoints();
    void restoreAdjoints();
    void registerAdjoints(const bool alwaysRegister);
    void registerAsOutput();
#endif
    void addCheckpoint();
    void replaceCheckpoint(int i);
    void restoreCheckpoint(int i);
    double getObjectSize();
    double calcNormOfStoredAdjoints();

  private:
    scalar& objRef;

    dco::mode<scalar>::value_t adjointStore;
    long int tapeIndexStore;
    std::vector<dco::mode<scalar>::value_t> checkpoints;
};

#include "CheckObjectScalar.C"
#endif // CHECKOBJECTSCALAR_H
