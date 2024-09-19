#ifndef CHECKOBJECT_H
#define CHECKOBJECT_H

#include "CheckBaseObject.H"
#include "GeometricField.H"

using namespace Foam;

template <class Type, template <class> class PatchField, class GeoMesh>
class CheckObjectGeometricField : public CheckBaseObject
{
    // resolver, doesn't need to define anything in it!
    template <class> struct resolver
    {
    }; // empty, yet powerful!

  public:
    // specializations for scalar and vector
    CheckObjectGeometricField(GeometricField<Type, PatchField, GeoMesh>* object,
                              bool reverseAccumulation) __attribute__((warn_unused_result));
#if defined(DAOF_AD_MODE_A1S)
    void storeAdjoints() { doStoreAdjoints(resolver<Type>()); }
    void restoreAdjoints()
    {
        if (adjointStore.size() > 0)
            doRestoreAdjoints(resolver<Type>());
    }
    void registerAdjoints(const bool alwaysRegister) { doRegisterAdjoints(resolver<Type>(),alwaysRegister); }
    void registerAsOutput() { doRegisterAsOutput(resolver<Type>()); }
    // void storeTapeIndex(){doRegisterAdjoints(resolver<Type>());}
#endif
    void addCheckpoint() { doAddCheckpoint(resolver<Type>()); }
    void replaceCheckpoint(int i) { doReplaceCheckpoint(resolver<Type>(), i); }
    void restoreCheckpoint(int i) { doRestoreCheckpoint(resolver<Type>(), i); }
    double getObjectSize();
    string name() const;

  private:
    GeometricField<Type, PatchField, GeoMesh>* objPtr;

    // std::vector<double> adjointStore;
    // typedef dco::trait_value<scalar>::RETURN_TYPE DCO_BASE_TYPE;
    typedef dco::mode<scalar>::value_t DCO_BASE_TYPE;
    std::vector<DCO_BASE_TYPE> adjointStore;
    std::vector<long int> tapeIndexStore;
    std::vector<std::vector<DCO_BASE_TYPE>> checkpoints;

    Foam::label nFieldValues;
    Foam::label nInternalFieldValues;
    bool reverseAccumulation;

#if defined(DAOF_AD_MODE_A1S)
    void doStoreAdjoints(const resolver<Foam::scalar>&);
    void doStoreAdjoints(const resolver<Foam::vector>&);

    double calcNormOfStoredAdjoints();

    void doRestoreAdjoints(const resolver<Foam::scalar>&);
    void doRestoreAdjoints(const resolver<Foam::vector>&);

    void doRegisterAdjoints(const resolver<Foam::scalar>&, const bool alwaysRegister);
    void doRegisterAdjoints(const resolver<Foam::vector>&, const bool alwaysRegister);

    void doRegisterAsOutput(const resolver<Foam::scalar>&);
    void doRegisterAsOutput(const resolver<Foam::vector>&);

    // void doStoreTapeIndex(const resolver<Foam::scalar>&);
    // void doStoreTapeIndex(const resolver<Foam::vector>&);
#endif

    void doAddCheckpoint(const resolver<Foam::scalar>&);
    void doAddCheckpoint(const resolver<Foam::vector>&);

    void doReplaceCheckpoint(const resolver<Foam::scalar>&, int);
    void doReplaceCheckpoint(const resolver<Foam::vector>&, int);

    void doRestoreCheckpoint(const resolver<Foam::scalar>&, int);
    void doRestoreCheckpoint(const resolver<Foam::vector>&, int);

    template <typename T = Type,
              std::enable_if_t<std::is_same<T, Foam::scalar>::value>* = nullptr>
    int getVecSize()
    {
        return 1;
    }
    template <typename T = Type,
              std::enable_if_t<std::is_same<T, Foam::vector>::value>* = nullptr>
    int getVecSize()
    {
        return 3;
    }
};
#include "CheckObjectGeometricField.C"
#endif // CHECKOBJECT_H
