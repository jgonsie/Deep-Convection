inline scalar Cv
(
    const scalar p,
    const scalar T
) const
{
    #if defined( __clang__ ) && !defined ( DAOF_MODE_ACTIVE )
    volatile const scalar cp = Cp(p, T);
    volatile const scalar cpmcv = EquationOfState::CpMCv(p, T);
    return cp - cpmcv;
    #else
    return Cp(p, T) - EquationOfState::CpMCv(p, T);
    #endif
}

inline scalar Es
(
    const scalar p,
    const scalar T
) const
{
    #if defined( __clang__ ) && !defined ( DAOF_MODE_ACTIVE )
    volatile const scalar hs = Hs(p, T);
    volatile const scalar rho = EquationOfState::rho(p, T);
    return hs - p/rho;
    #else
    return Hs(p, T) - p/EquationOfState::rho(p, T);
    #endif
}

inline scalar Ea
(
    const scalar p,
    const scalar T
) const
{
    #if defined( __clang__ ) && !defined ( DAOF_MODE_ACTIVE )
    volatile const scalar ha = Ha(p, T);
    volatile const scalar rho = EquationOfState::rho(p, T);
    return ha - p/rho;
    #else
    return Ha(p, T) - p/EquationOfState::rho(p, T);
    #endif
}
