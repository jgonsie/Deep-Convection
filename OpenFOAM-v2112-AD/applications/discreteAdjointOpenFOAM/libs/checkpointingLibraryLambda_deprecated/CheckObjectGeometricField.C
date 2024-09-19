//#include "CheckObjectGeometricField.H"
#include "fixedGradientFvPatchField.H"
#include "fixedFluxPressureFvPatchScalarField.H"
#include "mixedFvPatchField.H"

template <class Type, template <class> class PatchField, class GeoMesh>
CheckObjectGeometricField<Type, PatchField, GeoMesh>::CheckObjectGeometricField(
    GeometricField<Type, PatchField, GeoMesh>* object, bool reverseAccumulation)
    : objPtr(object), reverseAccumulation(reverseAccumulation)
{
    // const int dim = getVecSize(resolver<Type>());
    const int dim = getVecSize();
    nInternalFieldValues = dim * objPtr->field().size();
    nFieldValues = nInternalFieldValues;
    for (Foam::label i = 0; i < objPtr->boundaryField().size(); i++)
    {
        nFieldValues += dim * objPtr->boundaryField()[i].size();
        if(Foam::isA<const Foam::fixedGradientFvPatchScalarField>(objPtr->boundaryField()[i])){
            auto& bref = Foam::refCast<const Foam::fixedGradientFvPatchScalarField>(objPtr->boundaryField()[i]);
            nFieldValues += bref.gradient().size();
            std::cout << bref.type() << " "
                      << objPtr->name() << " "
                      << objPtr->mesh().boundary()[i].name() << " "
                      << " is  a fixedGradientFvPatchScalarField! "
                      << bref.gradient().size()
                      << std::endl;
        }
        if(Foam::isA<const Foam::mixedFvPatchField<Type>>(objPtr->boundaryField()[i])){
            auto& bref = Foam::refCast<const Foam::mixedFvPatchField<Type>>(objPtr->boundaryField()[i]);
            nFieldValues += bref.valueFraction().size();
            std::cout << bref.type() << " "
                      << objPtr->name() << " "
                      << objPtr->mesh().boundary()[i].name() << " "
                      << " is  a mixedFvPatchField! "
                      << std::endl;
        }
    }
}

#if defined(DAOF_AD_MODE_A1S)
template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doStoreAdjoints(
    const resolver<Foam::scalar>&)
{
#ifdef DEBUG_CHECKP
    Info << "scalar store adjoints " << objPtr->name() << endl;
#endif
    adjointStore.clear();
    tapeIndexStore.reserve(nFieldValues);

    // store adjoints of internal field and boundary field
    for (Foam::label i = 0; i < nFieldValues; i++)
    {
        adjointStore.push_back(ADmode::global_tape->_adjoint(tapeIndexStore[i]));
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doStoreAdjoints(
    const resolver<Foam::vector>&)
{
#ifdef DEBUG_CHECKP
    Info << "vector store adjoints " << objPtr->name() << endl;
#endif
    adjointStore.clear();
    tapeIndexStore.reserve(nFieldValues);

    for (Foam::label i = 0; i < nFieldValues; i++)
    {
        adjointStore.push_back(ADmode::global_tape->_adjoint(tapeIndexStore[i]));
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRestoreAdjoints(
    const resolver<Foam::scalar>& r)
{
#ifdef DEBUG_CHECKP
    Info << "scalar restore adjoints " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    if (!reverseAccumulation)
    {
        doRegisterAsOutput(r);
    }

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        AD::derivative(field[i]) = adjointStore[c];
        c++;
    }

    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            if (!reverseAccumulation || dco::tapeIndex(boundaryField[i][j]) != tapeIndexStore[c])
            {
                AD::derivative(boundaryField[i][j]) = adjointStore[c];
            }
            else
            {
                AD::derivative(boundaryField[i][j]) = 0;
            }
            c++;
        }
        // Checkpoint gradient if fixedGradientFvPatchField
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                AD::derivative(bref.gradient()[j]) = adjointStore[c++];
            }
        }
        // Checkpoint valueFraction if mixedFvPatchField
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                AD::derivative(bref.valueFraction()[j]) = adjointStore[c++];
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRestoreAdjoints(
    const resolver<Foam::vector>& r)
{
#ifdef DEBUG_CHECKP
    Info << "vector restore adjoints " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();
    if (!reverseAccumulation)
    {
        doRegisterAsOutput(r);
    }

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            AD::derivative(field[i][k]) = adjointStore[c];
            c++;
        }
    }

    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (!reverseAccumulation ||
                    dco::tapeIndex(boundaryField[i][j][k]) !=
                        tapeIndexStore[c])
                {
                    AD::derivative(boundaryField[i][j][k]) = adjointStore[c];
                }
                else
                {
                    AD::derivative(boundaryField[i][j][k]) = 0;
                }
                c++;
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                AD::derivative(bref.valueFraction()[j]) = adjointStore[c++];
            }
        }
    }
}

void registerInputVariable(scalar& x, const bool always_register){
    if(always_register || dco::tapeIndex(x) == 0){
        ADmode::global_tape->register_variable(x);
    }else{
        ADmode::global_tape->registerOutputVariable(x);
    }
}

void registerOutputVariable(scalar& x){
    if(dco::tapeIndex(x) == 0){
        ADmode::global_tape->register_variable(x);
    }else{
        ADmode::global_tape->registerOutputVariable(x);
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRegisterAdjoints(
    const resolver<Foam::scalar>&, const bool alwaysRegister)
{
#ifdef DEBUG_CHECKP
    Info << "scalar register adjoints " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    tapeIndexStore.clear();
    tapeIndexStore.reserve(nFieldValues);

    for (Foam::label i = 0; i < field.size(); i++)
    {
        registerInputVariable(field[i],alwaysRegister);
        tapeIndexStore.push_back(dco::tapeIndex(field[i]));
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            registerInputVariable(boundaryField[i][j],alwaysRegister);
            tapeIndexStore.push_back(dco::tapeIndex(boundaryField[i][j]));
        }
        // Checkpoint Gradient if fixedGradientFvPatchField
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            Foam::fixedGradientFvPatchScalarField& bref
                = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);

            forAll(bref.gradient(),j){
                registerInputVariable(bref.gradient()[j],true);
                tapeIndexStore.push_back(dco::tapeIndex(bref.gradient()[j]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);

            forAll(bref.valueFraction(),j){
                registerInputVariable(bref.valueFraction()[j],true);
                tapeIndexStore.push_back(dco::tapeIndex(bref.valueFraction()[j]));
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRegisterAdjoints(
    const resolver<Foam::vector>&, const bool alwaysRegister)
{
#ifdef DEBUG_CHECKP
    Info << "vector register adjoints " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    tapeIndexStore.clear();
    tapeIndexStore.reserve(nFieldValues);

    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            registerInputVariable(field[i][k],true);
            tapeIndexStore.push_back(dco::tapeIndex(field[i][k]));
        }
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                registerInputVariable(boundaryField[i][j][k],true);
                tapeIndexStore.push_back(dco::tapeIndex(boundaryField[i][j][k]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);

            forAll(bref.valueFraction(),j){
                registerInputVariable(bref.valueFraction()[j],true);
                tapeIndexStore.push_back(dco::tapeIndex(bref.valueFraction()[j]));
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRegisterAsOutput(
    const resolver<Foam::scalar>&)
{
#ifdef DEBUG_CHECKP
    Info << "scalar register as output " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    for (Foam::label i = 0; i < field.size(); i++)
    {
        registerOutputVariable(field[i]);
    }
    Foam::label c = nInternalFieldValues;
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            if (!reverseAccumulation || dco::tapeIndex(boundaryField[i][j]) != tapeIndexStore[c])
            {
                registerOutputVariable(boundaryField[i][j]);
            }
        }
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                registerOutputVariable(bref.gradient()[j]);
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                registerOutputVariable(bref.valueFraction()[j]);
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRegisterAsOutput(
    const resolver<Foam::vector>&)
{
#ifdef DEBUG_CHECKP
    Info << "vector register as output " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            registerOutputVariable(field[i][k]);
        }
    }
    Foam::label c = nInternalFieldValues;
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                if (!reverseAccumulation || dco::tapeIndex(boundaryField[i][j][k]) != tapeIndexStore[c])
                {
                    registerOutputVariable(boundaryField[i][j][k]);
                }
                c++;
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                registerOutputVariable(bref.valueFraction()[j]);
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
double
CheckObjectGeometricField<Type, PatchField, GeoMesh>::calcNormOfStoredAdjoints()
{
    double norm = 0;
    for (Foam::label i = 0;
         i < min(static_cast<Foam::label>(adjointStore.size()),
                 nInternalFieldValues);
         i++)
    {
        norm += adjointStore[i] * adjointStore[i];
    }
    return Foam::sqrt(norm);
}
#endif

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doAddCheckpoint(
    const resolver<Foam::scalar>&)
{
    //Info << "sens" << name() " has previousIterPtr: " << (fieldPrevIterPtr_ != 0) << endl;
#ifdef DEBUG_CHECKP
    Info << "scalar add checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    std::vector<DCO_BASE_TYPE> check;
    check.reserve(nFieldValues);

    for (Foam::label i = 0; i < field.size(); i++)
    {
        check.push_back(dco::value(field[i]));
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            check.push_back(dco::value(boundaryField[i][j]));
        }
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                check.push_back(dco::value(bref.gradient()[j]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                check.push_back(dco::value(bref.valueFraction()[j]));
            }
        }
    }
    checkpoints.push_back(check);
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doAddCheckpoint(
    const resolver<Foam::vector>&)
{
    //Info << "sens" << name() " has previousIterPtr: " << (fieldPrevIterPtr_ != 0) << endl;
#ifdef DEBUG_CHECKP
    Info << "vector add checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    std::vector<DCO_BASE_TYPE> check;
    check.reserve(nFieldValues);

    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            check.push_back(dco::value(field[i][k]));
        }
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                check.push_back(dco::value(boundaryField[i][j][k]));
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                check.push_back(dco::value(bref.valueFraction()[j]));
            }
        }
    }
    checkpoints.push_back(check);
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doReplaceCheckpoint(
    const resolver<Foam::scalar>&, int id)
{
    //Info << "sens" << name() " has previousIterPtr: " << (fieldPrevIterPtr_ != 0) << endl;
#ifdef DEBUG_CHECKP
    Info << "scalar replace checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        checkpoints[id][c++] = dco::value(field[i]);
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            checkpoints[id][c++] = dco::value(boundaryField[i][j]);
        }
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                checkpoints[id][c++] = dco::value(bref.gradient()[j]);
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                checkpoints[id][c++] = dco::value(bref.valueFraction()[j]);
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doReplaceCheckpoint(
    const resolver<Foam::vector>&, int id)
{
    //Info << "sens" << name() " has previousIterPtr: " << (fieldPrevIterPtr_ != 0) << endl;
#ifdef DEBUG_CHECKP
    Info << "vector replace checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            checkpoints[id][c] = dco::value(field[i][k]);
            c++;
        }
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                checkpoints[id][c] = dco::value(boundaryField[i][j][k]);
                c++;
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                checkpoints[id][c++] = dco::value(bref.valueFraction()[j]);
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRestoreCheckpoint(
    const resolver<Foam::scalar>&, int id)
{
#ifdef DEBUG_CHECKP
    Info << "scalar restore checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        field[i] = checkpoints[id][c];
        //dco::tapeIndex(field[i]) = tapeIndexStore[c];
        c++;
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            boundaryField[i][j] = checkpoints[id][c];
            //dco::tapeIndex(boundaryField[i][j]) = tapeIndexStore[c++];
            c++;
        }
        if(Foam::isA<Foam::fixedGradientFvPatchScalarField>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::fixedGradientFvPatchScalarField>(boundaryField[i]);
            forAll(bref.gradient(),j){
                bref.gradient()[j] = checkpoints[id][c];
                c++;
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<scalar>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<scalar>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                bref.valueFraction()[j] = checkpoints[id][c];
                c++;
            }
        }
    }
}

template <class Type, template <class> class PatchField, class GeoMesh>
void CheckObjectGeometricField<Type, PatchField, GeoMesh>::doRestoreCheckpoint(
    const resolver<Foam::vector>&, int id)
{
#ifdef DEBUG_CHECKP
    Info << "vector restore checkpoint " << objPtr->name() << endl;
#endif
    Foam::Field<Type>& field = objPtr->field();
    Foam::FieldField<PatchField, Type>& boundaryField =
        objPtr->boundaryFieldRef();

    Foam::label c = 0;
    for (Foam::label i = 0; i < field.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            field[i][k] = checkpoints[id][c];
            //dco::tapeIndex(field[i][k]) = tapeIndexStore[c];
            c++;
        }
    }
    for (Foam::label i = 0; i < boundaryField.size(); i++)
    {
        for (Foam::label j = 0; j < boundaryField[i].size(); j++)
        {
            for (int k = 0; k < 3; k++)
            {
                boundaryField[i][j][k] = checkpoints[id][c];
                //dco::tapeIndex(boundaryField[i][j][k]) = tapeIndexStore[c];
                c++;
            }
        }
        if(Foam::isA<Foam::mixedFvPatchField<vector>>(boundaryField[i])){
            auto& bref = Foam::refCast<Foam::mixedFvPatchField<vector>>(boundaryField[i]);
            forAll(bref.valueFraction(),j){
                bref.valueFraction()[j] = checkpoints[id][c];
                c++;
            }
        }
    }
}

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
