//#define DAOF_AD_MODE_A1S
#if defined(DAOF_AD_MODE_A1S)
#include <stack>

#include "CheckController.H"
#include "CheckDatabase.H"
#include "CheckRevolve.H"

CheckRevolve::CheckRevolve(CheckDatabase& cd, CheckDict& checkDict, Foam::Time& runTime)
    : 
    CheckMethod(cd, checkDict, runTime), 
    firstRun(true),
    lastInterpretation(false),
    nDoneTapeSteps(0),
    revolve(nCheckpoints) /* online checkpointing */
{
}

void CheckRevolve::run(CheckController& checkControl)
{
    auto startPosition = ADmode::global_tape->get_position();
    auto positionTapeSwitchOn = ADmode::global_tape->get_position(); // save position in order to interpret until here

    //std::stack<dco::ga1s<double>::tape_t::position_t> tapePositions;
    //std::stack<scalar> timeSteps;
    ADmode::global_tape->switch_to_passive();
    std::vector<double> adjointStore;

    enum ACTION::action whatodo;
    label nSteps = 0;
    do
    {
        whatodo = revolve.revolve();
        // store checkpoint and return
        if (whatodo == ACTION::takeshot)
        {
            Info << "CHECK: store Checkpoint " << revolve.getcheck() << " ("
                 << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                 << ")" << endl;
            checkData.replaceCheckpoint(revolve.getcheck());
        }
        // restore checkpoint and return
        else if (whatodo == ACTION::restore)
        {
            checkData.restoreCheckpoint(revolve.getcheck());
            Info << "CHECK: restore Checkpoint " << revolve.getcheck() << " ("
                 << runTime.timeIndex() << ", " << runTime.timeOutputValue()
                 << ")" << endl;
        }
        // run passively until it is time to switch on tape
        else if (whatodo == ACTION::advance)
        {
            int nAdvanceSteps = revolve.getcapo() - revolve.getoldcapo();
            Info << "CHECK: Advance passive from " << revolve.getoldcapo()
                 << " to " << revolve.getcapo() << endl;
            for (int i = 0; i < nAdvanceSteps; i++)
            {
                bool endReached = checkControl.runStep();
                nSteps++;
                if(endReached){
                    revolve.turn(nSteps);
                    firstRun = false;
                    break;
                }
            }
        }
        else if (whatodo == ACTION::firsturn) // adjoin cost functional back to state after last iteration
        {
            // calc cost and interpret
            ADmode::global_tape->switch_to_active();
            auto p = ADmode::global_tape->get_position();
            
            auto positionBeforeInputRegister = ADmode::global_tape->get_position();
            checkData.registerAdjoints();
            label dt = ADmode::global_tape->get_position().index() - positionBeforeInputRegister.index();
            adjointStore.resize(dt);
            scalar J = checkControl.calcCost();
            if(Pstream::master())
            {
                AD::derivative(J) = 1.0;
            }
            ADmode::global_tape->switch_to_passive();
            ADmode::global_tape->interpretTapeTo(p);

            checkData.storeAdjoints();
            /*for(int i=1; i<=dt; i++){
                adjointStore[i] = ADmode::global_tape->_adjoint(i+positionBeforeInputRegister.index());
            }*/
            
            ADmode::global_tape->reset_to(p);
        }
        // active section
        else if (whatodo == ACTION::youturn)
        {
            if (whatodo == ACTION::firsturn)
            {
                startPosition = ADmode::global_tape->get_position();
            }

            ADmode::global_tape->switch_to_active(); // switch on tape
            ADmode::global_tape->reset_to(startPosition);
            //auto positionBeforeInputRegister = ADmode::global_tape->get_position();
            checkData.registerAdjoints();
            //label dt = ADmode::global_tape->get_position().index() - positionBeforeInputRegister.index();

            positionTapeSwitchOn = ADmode::global_tape->get_position(); 

            Info << "CHECK: Advance active from " << revolve.getoldcapo()
                 << " to " << revolve.getoldcapo() + 1 << endl;

            checkControl.runStep();

            //auto positionBeforeOutputRegister = ADmode::global_tape->get_position();
            checkData.registerAsOutput();
            //label dt2 = ADmode::global_tape->get_position().index() - positionBeforeOutputRegister.index();

            checkData.restoreAdjoints();
            /*for(int i=1; i<=dt; i++){
                ADmode::global_tape->_adjoint(i+positionBeforeOutputRegister.index()) = adjointStore[i];
            }
            Info << "dt: " << dt << " " << dt2 << endl;*/

            Info << "Tape size: "  << (dco::size_of(ADmode::global_tape) / 1024.0 / 1024.0) << " MB" << endl;
            ADmode::global_tape->interpretTape_and_reset_to(positionTapeSwitchOn);
            checkControl.postInterpret();

            checkData.storeAdjoints(); // store adjoints
            /*for(int i=1; i<=dt; i++){
                adjointStore[i] = ADmode::global_tape->_adjoint(i+positionBeforeInputRegister.index());
            }*/
            ADmode::global_tape->switch_to_passive(); // switch off tape
        }
        if (whatodo == ACTION::error)
        {
            Info << " irregular termination of revolve " << endl;
        }
    } while((whatodo != ACTION::terminate) && (whatodo != ACTION::error));
}
#endif
