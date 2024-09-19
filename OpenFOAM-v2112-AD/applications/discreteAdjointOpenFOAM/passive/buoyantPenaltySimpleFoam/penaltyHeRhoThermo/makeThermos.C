#include "rhoThermo.H"
#include "makeThermo.H"

#include "specie.H"
#include "rhoConst.H"

#include "hConstThermo.H"
#include "penaltyHConstThermo.H"
#include "sensibleEnthalpy.H"
#include "sensibleInternalEnergy.H"
#include "thermo.H"

#include "constTransport.H"
#include "penaltyHeRhoThermo.H"
#include "pureMixture.H"


namespace Foam{
makeThermos
(
    rhoThermo,
    penaltyHeRhoThermo,
    pureMixture,
    constTransport,
    sensibleInternalEnergy,
    penaltyHConstThermo,
    rhoConst,
    specie
);

makeThermos
(
    rhoThermo,
    penaltyHeRhoThermo,
    pureMixture,
    constTransport,
    sensibleEnthalpy,
    penaltyHConstThermo,
    rhoConst,
    specie
);
}
