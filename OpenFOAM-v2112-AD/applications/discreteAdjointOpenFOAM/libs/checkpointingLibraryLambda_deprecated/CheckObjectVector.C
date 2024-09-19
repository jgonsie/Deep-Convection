//#include "CheckObjectScalar.H"

CheckObjectVector::CheckObjectVector(Foam::vector& object,
                                     bool reverseAccumulation)
    : objRef(object)
{
}

void CheckObjectVector::addCheckpoint()
{
    checkpoints.push_back(std::array<dco::mode<scalar>::value_t, 3>(
        {dco::value(objRef[0]), dco::value(objRef[1]), dco::value(objRef[2])}));
}

void CheckObjectVector::replaceCheckpoint(int i)
{
    checkpoints[i][0] = dco::value(objRef[0]);
    checkpoints[i][1] = dco::value(objRef[1]);
    checkpoints[i][2] = dco::value(objRef[2]);
}

void CheckObjectVector::restoreCheckpoint(int i)
{
    objRef[0] = checkpoints[i][0];
    objRef[1] = checkpoints[i][1];
    objRef[2] = checkpoints[i][2];
}

#if defined(DAOF_AD_MODE_A1S)
void CheckObjectVector::registerAdjoints(const bool alwaysRegister)
{
    ADmode::global_tape->register_variable(objRef[0]);
    tapeIndexStore = dco::tapeIndex(objRef);
    ADmode::global_tape->register_variable(objRef[1]);
    ADmode::global_tape->register_variable(objRef[2]);
}

void CheckObjectVector::registerAsOutput()
{
    ADmode::global_tape->registerOutputVariable(objRef[0]);
    ADmode::global_tape->registerOutputVariable(objRef[1]);
    ADmode::global_tape->registerOutputVariable(objRef[2]);
}

void CheckObjectVector::storeAdjoints()
{
    adjointStore = {ADmode::global_tape->_adjoint(tapeIndexStore + 0),
                    ADmode::global_tape->_adjoint(tapeIndexStore + 1),
                    ADmode::global_tape->_adjoint(tapeIndexStore + 2)};
}

void CheckObjectVector::restoreAdjoints()
{
    AD::derivative(objRef[0]) = adjointStore[0];
    AD::derivative(objRef[1]) = adjointStore[1];
    AD::derivative(objRef[2]) = adjointStore[2];
}
#endif

double CheckObjectVector::getObjectSize()
{
    return (sizeof(long int) + (checkpoints.size() + 1) * sizeof(double) * 3) /
           1024 / 1024; // return MB
}

double CheckObjectVector::calcNormOfStoredAdjoints()
{
    return std::sqrt(
        std::pow(AD::passiveValue(adjointStore[0]),2)
        + std::pow(AD::passiveValue(adjointStore[1]),2)
        + std::pow(AD::passiveValue(adjointStore[2]),2)
    );
}
