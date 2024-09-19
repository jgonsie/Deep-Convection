#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckNone.H"

CheckNone::CheckNone(CheckDatabase& cd, CheckDict& checkDict,
                     Foam::Time& runTime)
    : CheckMethod(cd, checkDict, runTime)
{
    startPositionSet = false;
}

void CheckNone::run(CheckController& checkControl)
{
    ADmode::global_tape->switch_to_active();
    std::stack<ADmode::tape_t::position_t> tapePositions;
    std::stack<scalar> timeSteps;

    tapePositions.push(ADmode::global_tape->get_position());
    //ADmode::global_tape->set_initial_barrier();
    Info << "myPos: " << ADmode::global_tape->get_position().index() << endl;
    //ADmode::global_tape->set_barrier(ADmode::tape_t::BARRIER_OVERWRITE);
    checkData.registerAdjoints(true);
    int dt = ADmode::global_tape->get_position().index() - tapePositions.top().index();
    std::cout << "DT: " << dt << std::endl;
    std::vector<double> adjointStore(dt,0.0);

    timeSteps.push(runTime.timeOutputValue());

    while (runTime.timeOutputValue() < runTime.endTime().value())
    {
        tapePositions.push(ADmode::global_tape->get_position());
        //ADmode::global_tape->set_barrier();
        checkData.registerAdjoints(true);
        checkControl.runStep();
        timeSteps.push(runTime.timeOutputValue());
        runTime.write();
        tapePositions.push(ADmode::global_tape->get_position());
        checkData.registerAsOutput();
    }

    scalar J = checkControl.calcCost();
    AD::derivative(J) = 1.0;

    while (tapePositions.size()>=2)
    {
        auto pos = tapePositions.top();
        tapePositions.pop();

        // restore adjoints
        for(int i=0; i<dt; i++){
            ADmode::global_tape->_adjoint(i+pos.index()) = adjointStore[i];
        }

        //checkControl.postInterpret();

        // store adjoints
        pos = tapePositions.top(); 
        tapePositions.pop();
        ADmode::global_tape->interpretTapeTo(pos);

        double sensSum = 0;        
        for(int i=0; i<dt; i++){
            sensSum += ADmode::global_tape->_adjoint(i+pos.index());
        }
        std::cout << "intermediate sensSum " << sensSum << std::endl;       

        // store adjoints
        for(int i=0; i<dt; i++){
            adjointStore[i] = ADmode::global_tape->_adjoint(i+pos.index());
        }

        ADmode::global_tape->reset_to(pos);
        runTime.setTime(timeSteps.top(), runTime.timeIndex() - 1);
        timeSteps.pop();

        scalar currentTime = runTime.timeOutputValue();
        scalar nearestTime = runTime.findClosestTime(currentTime).value();

        if (currentTime == 0 ||
            abs(nearestTime - currentTime) < doubleScalarSMALL)
        {
            checkControl.write(false);
        }
    }
    auto pos = tapePositions.top(); 
    tapePositions.pop();

    // restore adjoints
    for(int i=0; i<dt; i++){
        ADmode::global_tape->_adjoint(i+pos.index()) += adjointStore[i];
    }
}
#endif
