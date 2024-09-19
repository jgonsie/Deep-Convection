//#include "CheckObjectDoubleScalar.H"

CheckObjectDoubleScalar::CheckObjectDoubleScalar(scalar &object){
    objPtr = &object;
}

void CheckObjectDoubleScalar::addCheckpoint(){
    checkpoints.push_back(objPtr->_value());
}

void CheckObjectDoubleScalar::replaceCheckpoint(int i){
    checkpoints[i] = objPtr->_value();
}

void CheckObjectDoubleScalar::restoreCheckpoint(int i){
    *objPtr = checkpoints[i];
}

void CheckObjectDoubleScalar::registerAdjoints(){
    ADmode::global_tape->register_variable(*objPtr);
    tapeIndexStore = objPtr->data().tapeIndex();
}

void CheckObjectDoubleScalar::storeAdjoints(){
    double tmp;
    ADmode::get(*objPtr,tmp,-1);
    adjointStore = tmp;
}

void CheckObjectDoubleScalar::restoreAdjoints(){
    double tmp;
    ADmode::global_tape->registerOutputVariable(*objPtr);
    ADmode::set(*objPtr,adjointStore,-1);
}

double CheckObjectDoubleScalar::getObjectSize(){
    return (sizeof(long int) + (checkpoints.size() + 1)*sizeof(double)) / 1024 / 1024; // return MB
}

double calcNormOfStoredAdjoints(){
    return sqrt(adjointStore*adjointStore);
}
