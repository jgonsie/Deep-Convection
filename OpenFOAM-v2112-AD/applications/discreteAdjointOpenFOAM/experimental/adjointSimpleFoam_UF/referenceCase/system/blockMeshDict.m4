/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  1.7.1                                 |
|   \\  /    A nd           | Web:      www.OpenFOAM.com                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

scale 1;

vertices
(
  (0 0 0) (2 0 0) (4 0 0) (5 0 0) (0 1 0) (2 1 0) (4 1 0) 
  (5 1 0) (2 3 0) (4 3 0) (5 3 0) (4 5 0) (5 5 0)
  (0 0 1) (2 0 1) (4 0 1) (5 0 1) (0 1 1) (2 1 1) (4 1 1)
  (5 1 1) (2 3 1) (4 3 1) (5 3 1) (4 5 1) (5 5 1)
);

m4_define(lvl,1)
blocks          
(
    hex (0 1 5 4 13 14 18 17)    ( m4_eval(lvl*2)  m4_eval(lvl*1)  1 ) simpleGrading (1 1 1)
    hex (1 2 6 5 14 15 19 18)    ( m4_eval(lvl*2)  m4_eval(lvl*1)  1 ) simpleGrading (1 1 1)
    hex (2 3 7 6 15 16 20 19)    ( m4_eval(lvl*1)   m4_eval(lvl*1)  1 ) simpleGrading (1 1 1)
    hex (5 6 9 8 18 19 22 21)    ( m4_eval(lvl*2)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (6 7 10 9 19 20 23 22)   ( m4_eval(lvl*1)   m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (9 10 12 11 22 23 25 24) ( m4_eval(lvl*1)   m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
);

patches         
(
    patch inlet
    (
        (0 4 17 13)
    )    
    wall walls 
    (
        (0 1 14 13) (1  2 15 14) ( 2  3 16 15) (4 5 18 17)
        (3 7 20 16) (7 10 23 20) (10 12 25 23)
        (5 8 21 18) (8  9 22 21) ( 9 11 24 22)
    )
    patch outlet
    (
        (11 12 25 24)
    )
        
);

// ************************************************************************* //
