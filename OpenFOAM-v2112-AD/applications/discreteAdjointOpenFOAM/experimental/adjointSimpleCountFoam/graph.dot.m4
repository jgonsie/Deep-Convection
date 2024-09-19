digraph finite_state_machine {
    {rank = min; U_in p_in phi_in}
    {rank = max; U_out p_out phi_out}

    UEqn     [label="UEqn\n m4_UEqn"]
    fullUEqn [label="fullUEqn\n m4_fullUEqn"]
    solveU   [label="solveU\n m4_solveU"]
    rAU      [label="rAU\n m4_rAU"]
    HbyA     [label="HbyA\n m4_HbyA"]
    phiHbyA  [label="phiHbyA\n m4_phiHbyA"]
    pEqn     [label="pEqn\n m4_pEqn"]
    solveP   [label="solveP\n m4_solveP"]
    phi1     [label="phi1\n m4_phi1"]
    U2       [label="U2\n m4_U2"]
    
    U_in -> UEqn phi_in -> UEqn
    UEqn -> fullUEqn p_in -> fullUEqn
    fullUEqn -> solveU
    UEqn -> rAU
    rAU -> HbyA UEqn -> HbyA solveU -> HbyA p_in -> HbyA
    HbyA -> phiHbyA solveU -> phiHbyA p_in -> phiHbyA
    rAU -> pEqn solveU -> pEqn phiHbyA -> pEqn
    pEqn -> solveP
    phiHbyA -> phi1 pEqn -> phi1
    HbyA -> U2 rAU -> U2 solveP -> U2
    U2 -> U_out
    solveP -> p_out
    phi1 -> phi_out
}