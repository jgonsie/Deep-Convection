#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckDebug.H"

CheckDebug::CheckDebug(CheckDatabase& cd, CheckDict& checkDict,
                     Foam::Time& runTime)
    : CheckMethod(cd, checkDict, runTime)
{
    startPositionSet = false;
}

void CheckDebug::run(CheckController& checkControl)
{
    ADmode::global_tape->switch_to_active();
    std::stack<ADmode::tape_t::position_t> tapePositions;
    std::stack<scalar> timeSteps;

    tapePositions.push(ADmode::global_tape->get_position());
    //ADmode::global_tape->set_initial_barrier();
    checkData.registerAsOutput();
    int dt = ADmode::global_tape->get_position().index() - tapePositions.top().index();
    std::cout << "DT: " << dt << std::endl;
    std::vector<double> adjointStore(dt,0.0);

    timeSteps.push(runTime.timeOutputValue());

    while (runTime.timeOutputValue() < runTime.endTime().value())
    {
        //ADmode::global_tape->set_barrier();
        tapePositions.push(ADmode::global_tape->get_position());
        checkData.registerAdjoints(false);
        checkControl.runStep();
        timeSteps.push(runTime.timeOutputValue());
        runTime.write();
        checkData.registerAsOutput();
        //tapePositions.push(ADmode::global_tape->get_position());
    }

    scalar J = checkControl.calcCost();
    AD::derivative(J) = 1.0;

    while (!tapePositions.empty())
    {
        // store adjoints
        auto pos = tapePositions.top();
        tapePositions.pop();

        ADmode::global_tape->interpretTapeTo(pos);

        double sensSum = 0;
        for(int i=0; i<dt; i++){
            sensSum += ADmode::global_tape->_adjoint(i+pos.index());
        }
        std::cout << "intermediate sensSum " << sensSum << std::endl;

        // store adjoints
        //for(int i=0; i<dt; i++){
        //    adjointStore[i] = ADmode::global_tape->_adjoint(i+pos.index());
        //}

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
    tapePositions.pop();

    // restore adjoints
    //for(int i=0; i<dt; i++){
    //    ADmode::global_tape->_adjoint(i+pos.index()) += adjointStore[i];
    //}
}
#endif
