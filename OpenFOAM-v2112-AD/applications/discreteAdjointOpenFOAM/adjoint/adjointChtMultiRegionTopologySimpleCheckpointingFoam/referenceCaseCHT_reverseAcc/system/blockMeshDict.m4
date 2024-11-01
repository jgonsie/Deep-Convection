
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

verbose no;
scale 1;

vertices
(
  (-1 0 0)      // 0
  (1.5 0 0)     // 1
  (2 0 0)       // 2
  (4 0 0)       // 3
  (5 0 0)       // 4
  (5.5 0 0)     // 5
  (-1 1 0)      // 6
  (1.5 1 0)     // 7
  (2 1 0)       // 8
  (4 1 0)       // 9
  (5 1 0)       // 10
  (5.5 1 0)     // 11
  (1.5 3 0)     // 12
  (2 3 0)       // 13
  (4 3 0)       // 14
  (5 3 0)       // 15
  (5.5 3 0)     // 16
  (1.5 3.5 0)   // 17
  (2 3.5 0)     // 18
  (4 3.5 0)     // 19
  (5 3.5 0)     // 20
  (4 6 0)       // 21
  (5 6 0)       // 22
  (2 -0.5 0)    // 23
  (4 -0.5 0)    // 24
  (5 -0.5 0)    // 25
  (5.5 -0.5 0)  // 26
  (-1 0 0.1)
  (1.5 0 0.1)
  (2 0 0.1)
  (4 0 0.1)
  (5 0 0.1)
  (5.5 0 0.1)
  (-1 1 0.1)
  (1.5 1 0.1)
  (2 1 0.1)
  (4 1 0.1)
  (5 1 0.1)
  (5.5 1 0.1)
  (1.5 3 0.1)
  (2 3 0.1)
  (4 3 0.1)
  (5 3 0.1)
  (5.5 3 0.1)
  (1.5 3.5 0.1)
  (2 3.5 0.1)
  (4 3.5 0.1)
  (5 3.5 0.1)
  (4 6 0.1)
  (5 6 0.1)
  (2 -0.5 0.1)
  (4 -0.5 0.1)
  (5 -0.5 0.1)
  (5.5 -0.5 0.1)
);

m4_define(lvl,1)
blocks
(
    hex (0 1 7 6 27 28 34 33)     fluid  ( m4_eval(lvl*5)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (1 2 8 7 28 29 35 34)     fluid  ( m4_eval(lvl*1)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (2 3 9 8 29 30 36 35)     fluid  ( m4_eval(lvl*3)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (3 4 10 9 30 31 37 36)    fluid  ( m4_eval(lvl*2)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (8 9 14 13 35 36 41 40)   fluid  ( m4_eval(lvl*3)  m4_eval(lvl*3) 1 ) simpleGrading (1 1 1)
    hex (9 10 15 14 36 37 42 41)  fluid  ( m4_eval(lvl*2)  m4_eval(lvl*3) 1 ) simpleGrading (1 1 1)
    hex (14 15 20 19 41 42 47 46) fluid  ( m4_eval(lvl*2)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
    hex (19 20 22 21 46 47 49 48) fluid  ( m4_eval(lvl*2)  m4_eval(lvl*5) 1 ) simpleGrading (1 1 1)
//  solid top left
    hex (7 8 13 12 34 35 40 39)   solid ( m4_eval(lvl*1)  m4_eval(lvl*3) 1 ) simpleGrading (1 1 1)
    hex (12 13 18 17 39 40 45 44) solid ( m4_eval(lvl*1)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
    hex (13 14 19 18 40 41 46 45) solid ( m4_eval(lvl*3)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
// solid bottom right
    hex (23 24 3 2 50 51 30 29)   solid ( m4_eval(lvl*3)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
    hex (24 25 4 3 51 52 31 30)   solid ( m4_eval(lvl*2)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
    hex (25 26 5 4 52 53 32 31)   solid ( m4_eval(lvl*1)  m4_eval(lvl*1) 1 ) simpleGrading (1 1 1)
    hex (4 5 11 10 31 32 38 37)   solid ( m4_eval(lvl*1)  m4_eval(lvl*2) 1 ) simpleGrading (1 1 1)
    hex (10 11 16 15 37 38 43 42) solid ( m4_eval(lvl*1)  m4_eval(lvl*3) 1 ) simpleGrading (1 1 1)
);

defaultPatch
{
    name    front_and_back;
    type    empty;
}

patches
(
    patch inlet
    (
      (0 6 33 27)
    )

    patch outlet
    (
      (22 21 49 48)
    )

    patch solid_heated_wall_top
    (
      (7 12 39 34)
      (12 17 44 39)
      (17 18 45 44)
      (18 19 46 45)
    )

    patch solid_heated_wall_bottom
    (
      (23 24 51 50)
      (24 25 52 51)
      (25 26 53 52)
    )

    patch fluid_walls
    (
      (0 1 28 27)
      (1 2 29 28)
      (7 6 33 34)
      //(2 3 30 29)
      //(3 4 31 30)
      //(4 10 37 31)
      //(10 15 42 37)
      (15 20 47 42)
      (20 22 49 47)
      (21 19 46 48)
    )
);

// ************************************************************************* //
