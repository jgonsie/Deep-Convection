#ifndef CHECKOBJECTVECTOR_H
#define CHECKOBJECTVECTOR_H
#include "CheckBaseObject.H"
#include <array>

class CheckObjectVector : public CheckBaseObject
{
  public:
    // specializations for scalar and vector
    CheckObjectVector(Foam::vector& object, bool reverseAccumulation = false);
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
    Foam::vector& objRef;

    std::array<AD_BASE_TYPE, 3> adjointStore;
    long int tapeIndexStore;
    std::vector<std::array<AD_BASE_TYPE, 3>> checkpoints;
};

#include "CheckObjectVector.C"
#endif // CHECKOBJECTVECTOR_H
