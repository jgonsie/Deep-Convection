#ifndef CHECKOBJECTDICTIONARY_H
#define CHECKOBJECTDICTIONARY_H
#include "CheckBaseObject.H"
#include "Dictionary.H"
class CheckObjectDictionary : public CheckBaseObject
{
  public:
    // specializations for scalar and vector
    CheckObjectDictionary(Foam::dictionary& object, bool reverseAccumulation = false);
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
    Foam::dictionary& objRef;
    std::vector<Foam::dictionary> checkpoints;
};

#include "CheckObjectDictionary.C"
#endif // CHECKOBJECTDICTIONARY_H
