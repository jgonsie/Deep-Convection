#ifndef CHECKOBJECT_H
#define CHECKOBJECT_H

#include "CheckBaseObject.H"
#include "GeometricField.H"

#include <type_traits>
#include <cstddef>

using namespace Foam;

template <class Type, template <class> class PatchField, class GeoMesh>
class CheckObjectGeometricField : public CheckBaseObject
{
  public:
    // specializations for scalar and vector
    CheckObjectGeometricField(GeometricField<Type, PatchField, GeoMesh>* object,
                              bool reverseAccumulation);
#if defined(DAOF_AD_MODE_A1S)
    void storeAdjoints();
    void restoreAdjoints();
    void registerAdjoints();
    void registerAsOutput();
#endif
    std::vector<AD_BASE_TYPE> createCheckpoint();
    void addCheckpoint();
    void replaceCheckpoint(int i);
    void restoreCheckpoint(int i);
    double getObjectSize();
    string name() const;

  private:
    label dim;
    GeometricField<Type, PatchField, GeoMesh>* objPtr;

    std::vector<AD_BASE_TYPE> adjointStore;
    std::vector<long int> tapeIndexStore;
    std::vector<std::vector<AD_BASE_TYPE>> checkpoints;

    Foam::label nFieldValues;
    Foam::label nInternalFieldValues;
    bool reverseAccumulation;

#if defined(DAOF_AD_MODE_A1S)
    double calcNormOfStoredAdjoints();

#endif

    int getVecSize();
};
#include "CheckObjectGeometricField.C"
#endif // CHECKOBJECT_H
