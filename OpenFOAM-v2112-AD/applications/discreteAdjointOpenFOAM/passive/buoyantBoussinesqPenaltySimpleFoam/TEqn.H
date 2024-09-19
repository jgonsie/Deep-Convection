{
    alphat = turbulence->nut()/Prt;
    alphat.correctBoundaryConditions();
    
    volScalarField alphaEff("alphaEff", penalty*alphaSolid+(1-penalty)*(turbulence->nu()/Pr) + alphat);

    fvScalarMatrix TEqn
    (
        fvm::div(phi, T)
      - fvm::laplacian(alphaEff, T)
     ==
        radiation->ST(rhoCpRef, T)
      + fvOptions(T)
    );

    TEqn.relax();

    fvOptions.constrain(TEqn);

    TEqn.solve();

    radiation->correct();

    fvOptions.correct(T);

    rhok = 1.0 - beta*(T - TRef);
}
