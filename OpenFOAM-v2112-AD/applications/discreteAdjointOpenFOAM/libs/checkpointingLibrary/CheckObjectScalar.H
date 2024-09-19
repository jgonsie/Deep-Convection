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
    void registerAdjoints();
    void registerAsOutput();
#endif
    void addCheckpoint();
    void replaceCheckpoint(int i);
    void restoreCheckpoint(int i);
    double getObjectSize();
    double calcNormOfStoredAdjoints();

  private:
    scalar& objRef;

    AD_BASE_TYPE adjointStore;
    long int tapeIndexStore;
    std::vector<AD_BASE_TYPE> checkpoints;
};

#include "CheckObjectScalar.C"
#endif // CHECKOBJECTSCALAR_H
