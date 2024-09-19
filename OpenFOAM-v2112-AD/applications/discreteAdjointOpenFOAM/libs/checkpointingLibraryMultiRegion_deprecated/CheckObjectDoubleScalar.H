#ifndef CHECKOBJECTDOUBLESCLAR_H
#define CHECKOBJECTDOUBLESCLAR_H
#include "CheckBaseObject.H"

class CheckObjectDoubleScalar : public CheckBaseObject
{
public:
    // specializations for scalar and vector
    CheckObjectDoubleScalar(Foam::scalar &object, bool reverseAccumulation = false);
    void storeAdjoints();
    void restoreAdjoints();
    void registerAdjoints();
    void addCheckpoint();
    void replaceCheckpoint(int i);
    void restoreCheckpoint(int i);
    double getObjectSize();
    double calcNormOfStoredAdjoints();

private:
    scalar* objPtr;

    double adjointStore;
    long int tapeIndexStore;
    std::vector< double > checkpoints;
};

#include "CheckObjectDoubleScalar.C"
#endif // CHECKOBJECTDOUBLESCLAR_H
