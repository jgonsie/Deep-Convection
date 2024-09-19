//#include "CheckObjectGeometricField.H"
#include "fixedGradientFvPatchField.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "mixedFvPatchField.H"

#include <cassert>

template <class Type, template <class> class PatchField, class GeoMesh>
CheckObjectGeometricField<Type, PatchField, GeoMesh>::CheckObjectGeometricField(
    GeometricField<Type, PatchField, GeoMesh>* object, bool reverseAccumulation)
    : dim(getVecSize()), objPtr(object), reverseAccumulation(reverseAccumulation)
{
    nInternalFieldValues = dim * objPtr->field().size();

    nFieldValues = nInternalFieldValues;
    for (Foam::label i = 0; i < objPtr->boundaryField().size(); i++)
    {
        nFieldValues += dim * objPtr->boundaryField()[i].size();
        if(Foam::isA<const Foam::fixedGradientFvPatchScalarField>(objPtr->boundaryField()[i])){
            const auto& bref = Foam::refCast<const Foam::fixedGradientFvPatchScalarField>(objPtr->boundaryField()[i]);
            nFieldValues += bref.gradient().size();
        }
        if(Foam::isA<const Foam::mixedFvPatchField<Type>>(objPtr->boundaryField()[i])){
            const auto& bref = Foam::refCast<const Foam::mixedFvPatchField<Type>>(objPtr->boundaryField()[i]);
            nFieldValues += bref.valueFraction().size();
        }
    }
}

#if defined(DAOF_AD_MODE_A1S)
template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::storeAdjoints()
{
    adjointStore.resize(nFieldValues);

    // store adjoints of internal field and boundary field
    for (Foam::label i = 0; i < nFieldValues; i++)
    {
        adjointStore[i] = AD::adjointFromIndex(tapeIndexStore[i]);
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::restoreAdjoints()
{
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField = objPtr->boundaryFieldRef();
    scalar* scalarPtr = reinterpret_cast<scalar*>(field.data());

    if (!reverseAccumulation)
    {
        registerAsOutput();
    }

    Foam::label c = 0;
    for (Foam::label i = 0; i < nInternalFieldValues; i++)
    {
        AD::derivative(scalarPtr[i]) = adjointStore[c++];
    }

    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        scalar* scalarBoundaryPtr = reinterpret_cast<scalar*>(boundaryField[i].data());
        for (Foam::label j = 0; j < dim*boundaryField[i].size(); j++)
        {
            if (!reverseAccumulation || AD::tapeIndex(scalarBoundaryPtr[j]) != tapeIndexStore[c])
            {
                AD::derivative(scalarBoundaryPtr[j]) = adjointStore[c];
            }
            else
            {
                AD::derivative(scalarBoundaryPtr[j]) = 0;
            }
            c++;
        }

        // Checkpoint Gradient if fixedGradientFvPatchField
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]))
        {
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(), j)
            {
                AD::derivative(bref.gradient()[j]) = adjointStore[c++];
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<Type>>(boundaryField[i]))
        {
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<Type>>(boundaryField[i]);
            forAll(bref.valueFraction(), j)
            {
                AD::derivative(bref.valueFraction()[j]) = adjointStore[c++];
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::registerAdjoints()
{
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    scalar* scalarPtr = reinterpret_cast<scalar*>(field.data());

    tapeIndexStore.clear();
    tapeIndexStore.reserve(nFieldValues);

    for (Foam::label i = 0; i < nInternalFieldValues; i++)
    {
        AD::registerInputVariable(scalarPtr[i]);
        tapeIndexStore.push_back(AD::tapeIndex(scalarPtr[i]));
    }

    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        scalar* scalarBoundaryPtr = reinterpret_cast<scalar*>(boundaryField[i].data());
        label nValues = boundaryField[i].size_bytes() / sizeof(scalar);

        for (Foam::label j = 0; j < nValues; j++)
        {
            AD::registerInputVariable(scalarBoundaryPtr[j]);
            tapeIndexStore.push_back(AD::tapeIndex(scalarBoundaryPtr[j]));
        }

        // Checkpoint Gradient if fixedGradientFvPatchField
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]))
        {
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                AD::registerInputVariable(bref.gradient()[j]);
                tapeIndexStore.push_back(AD::tapeIndex(bref.gradient()[j]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<Type>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<Type>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                AD::registerInputVariable(bref.valueFraction()[j]);
                tapeIndexStore.push_back(AD::tapeIndex(bref.valueFraction()[j]));
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::registerAsOutput()
{
#ifdef DEBUG_CHECKP
    Info << "register as output " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    scalar* scalarPtr = reinterpret_cast<scalar*>(field.data());

    for (Foam::label i = 0; i < nInternalFieldValues; i++)
    {
        AD::registerOutputVariable(scalarPtr[i]);
    }
    
    Foam::label c = nInternalFieldValues;
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        scalar* scalarBoundaryPtr = reinterpret_cast<scalar*>(boundaryField[i].data());
        label nValues = boundaryField[i].size_bytes() / sizeof(scalar);

        for (Foam::label j = 0; j < nValues; j++)
        {
            if (!reverseAccumulation || AD::tapeIndex(scalarBoundaryPtr[j]) != tapeIndexStore[c])
            {
                AD::registerOutputVariable(scalarBoundaryPtr[j]);
            }
        }
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]))
        {
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j)
            {
                AD::registerOutputVariable(bref.gradient()[j]);
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<Type>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<Type>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                AD::registerOutputVariable(bref.valueFraction()[j]);
            }
        }
    }
}
#endif

template <class Type, template <class> class PatchField, class GeoMesh>
std::vector<double> CheckObjectGeometricField<Type, PatchField, GeoMesh>::createCheckpoint()
{
#ifdef DEBUG_CHECKP
    Info << "add checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    scalar* scalarPtr = reinterpret_cast<scalar*>(field.data());

    std::vector<AD_BASE_TYPE> check;
    check.reserve(nFieldValues);

    for (Foam::label i = 0; i < nInternalFieldValues; i++)
    {
        check.push_back(AD::value(scalarPtr[i]));
    }

    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        scalar* scalarBoundaryPtr = reinterpret_cast<scalar*>(boundaryField[i].data());
        for (Foam::label j = 0; j < dim*boundaryField[i].size(); j++)
        {
            check.push_back(AD::value(scalarBoundaryPtr[j]));
        }

        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            const auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                check.push_back(AD::value(bref.gradient()[j]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<Type>>(boundaryField[i])){
            const auto& bref = Foam::refCast<Foam::mixedFvPatchField<Type>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                check.push_back(AD::value(bref.valueFraction()[j]));
            }
        }
    }
    return check;
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::addCheckpoint()
{
    checkpoints.push_back(createCheckpoint());
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::replaceCheckpoint(int id)
{
#ifdef DEBUG_CHECKP
    Info << "replace checkpoint " << objPtr->name() << endl;
#endif
    assert(int(checkpoints.size()) > id);
    checkpoints[id] = createCheckpoint();
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::restoreCheckpoint(int id)
{
#ifdef DEBUG_CHECKP
    Info << "restore checkpoint " << objPtr->name() << endl;
#endif
    std::vector<AD_BASE_TYPE>& checkpoint = checkpoints[id];
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    scalar* scalarPtr = reinterpret_cast<scalar*>(field.data());

    Foam::label c = 0;
    for (Foam::label i = 0; i < dim*field.size(); i++)
    {
        scalarPtr[i] = checkpoint[c++];
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        scalar* scalarBoundaryPtr = reinterpret_cast<scalar*>(boundaryField[i].data());
        label nValues = boundaryField[i].size_bytes() / sizeof(scalar);
        for (Foam::label j = 0; j < nValues; j++)
        {
            scalarBoundaryPtr[j] = checkpoint[c++];
        }

        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                bref.gradient()[j] = checkpoint[c++];
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<Type>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<Type>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                bref.valueFraction()[j] = checkpoint[c++];
            }
        }
    }
}

#if defined(DAOF_AD_MODE_A1S)
template <class Type, template <class> class PatchField, class GeoMesh>
double
CheckObjectGeometricField<Type, PatchField, GeoMesh>::calcNormOfStoredAdjoints()
{
    double norm = 0;
    for (unsigned int i = 0; i < adjointStore.size(); i++)
    {
        norm += adjointStore[i] * adjointStore[i];
    }
    return Foam::sqrt(norm);
}
#endif

template <class Type, template <class> class PatchField, class GeoMesh>
string CheckObjectGeometricField<Type, PatchField, GeoMesh>::name() const
{
    return objPtr->name();
}

template <class Type, template <class> class PatchField, class GeoMesh>
double CheckObjectGeometricField<Type, PatchField, GeoMesh>::getObjectSize()
{
    // const int vecSize = getVecSize(resolver<Type>());
    const Foam::label vecSize = getVecSize();
    const int doubleSize = sizeof(double);
    const int intSize = sizeof(long int);
    double memSize =
        nFieldValues * vecSize *
        (intSize + doubleSize +
         checkpoints.size() *
             doubleSize);         // tape indices + adjoint store + checkpoints
    return memSize / 1024 / 1024; // return MB
}

template <class Type, template <class> class PatchField, class GeoMesh>
int CheckObjectGeometricField<Type, PatchField, GeoMesh>::getVecSize()
{
    if constexpr(std::is_same_v<Type,scalar>)
    {
        return 1;
    }
    else
    {
        return 3;
    }
}
