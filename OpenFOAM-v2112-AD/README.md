# Custom Discrete Adjoint OpenFOAM
Discrete Adjoint OpenFOAM is a version of OpenFOAM which has Algorithmic Differentiation by Operator Overloading applied to the whole OpenFOAM core package.
For references visit the [STCE homepage](https://www.stce.rwth-aachen.de/research/software/discreteadjointopenfoam). The original code may be found in the [Markus Towara's repository](https://gitlab.stce.rwth-aachen.de/towara/discreteadjointopenfoam_adwrapper/-/tree/master).

Particularly, this repository contains a custom version of the Discrete Adjoint OpenFOAM, which was modified to couple the original code with the deep learning framework. The modifications does not affect the normal behaviour of the original code.

## How to install this version of OpenFOAM?
The procedure for the installation of this OpenFOAM version is the same than for any other oficial version.
1. Download the source code from this repository and extract it on a directory, e.g. `home/myuser/OpenFOAM`.
2. Check [prerequisites](https://www.openfoam.com/documentation/system-requirements) for installing OpenFOAM.
3. Source the OpenFOAM's bashrc file, by adding the following line to the user bashrc (located in `home/myuser/.bashrc`):
   ```
   source ~/OpenFOAM/OpenFOAM-v2112-AD/etc/bashrc
   ```
4. Compile the source code by the following command:
   ```
   ./Allwmake -j
   ```

## What modifications does the custom verstion have?
We added two main modifications:
1. Communication by RAM between Python and OpenFOAM (c++). We modified the reading of the fields involved in the simulation.
2. Acceptance of the Ufaces field by modifiying the classes `src/finiteVolume/finiteVolume/fvm/fvmDiv` and `src/finiteVolume/convectionSchemes/`.
   
## Contact Discreate Adjoint OpenFOAM developers
towara@stce.rwth-aachen.de

## Disclaimer
This offering is not approved or endorsed by OpenCFD Limited, producer and distributor of the OpenFOAM software and owner of the OPENFOAM®  and OpenCFD®  trade marks.

## About OpenFOAM
OpenFOAM is a free, open source CFD software [released and developed primarily by OpenCFD Ltd](http://www.openfoam.com) since 2004. It has a large user base across most areas of engineering and science, from both commercial and academic organisations. OpenFOAM has an extensive range of features to solve anything from complex fluid flows involving chemical reactions, turbulence and heat transfer, to acoustics, solid mechanics and electromagnetics.  [More...](http://www.openfoam.com/documentation)


OpenFOAM is professionally released every six months to include customer sponsored developments and contributions from the community - individual and group contributors, fork re-integrations including from FOAM-extend and OpenFOAM Foundation Ltd - in this Official Release sanctioned by the OpenFOAM Worldwide Trademark Owner aiming towards one OpenFOAM.


## Copyright
OpenFOAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  See the file COPYING in this directory or [http://www.gnu.org/licenses/](http://www.gnu.org/licenses), for a description of the GNU General Public License terms under which you can copy the files.


## OpenFOAM Trademark
OpenCFD Ltd grants use of its OpenFOAM trademark by Third Parties on a licence basis. ESI Group and OpenFOAM Foundation Ltd are currently permitted to use the Name and agreed Domain Name. For information on trademark use, please refer to the [trademark policy guidelines](http://www.openfoam.com/legal/trademark-policy.php).

Please [contact OpenCFD](http://www.openfoam.com/contact) if you have any questions on the use of the OpenFOAM trademark.

Violations of the Trademark are continuously monitored, and will be duly prosecuted.


# Useful Links
- [Download and installation instructions](http://www.openfoam.com/download/)
- [Documentation](http://www.openfoam.com/documentation)
- [Reporting bugs/issues/feature requests](http://www.openfoam.com/code/bug-reporting.php)
- [OpenFOAM Community](http://www.openfoam.com/community/)
- [Contacting OpenCFD](http://www.openfoam.com/contact/)

Copyright 2016-2025 OpenCFD Ltd


## About OpenFOAM
OpenFOAM is a free, open source CFD software [released and developed by OpenCFD Ltd since 2004](http://www.openfoam.com/history/).
It has a large user base across most areas of engineering and science, from both commercial and academic organisations.
OpenFOAM has an extensive range of features to solve anything from complex fluid flows involving chemical reactions, turbulence and heat transfer, to acoustics, solid mechanics and electromagnetics.
[See documentation](http://www.openfoam.com/documentation)

OpenFOAM is professionally released every six months to include
customer sponsored developments and contributions from the community -
individual and group contributors, integrations
(eg, from FOAM-extend and OpenFOAM Foundation Ltd) as well as
[governance guided activities](https://www.openfoam.com/governance/).


## License

OpenFOAM is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.  See the file COPYING in this directory or
[http://www.gnu.org/licenses/](http://www.gnu.org/licenses), for a
description of the GNU General Public License terms under which you
may redistribute files.


## OpenFOAM Trademark

OpenCFD Ltd grants use of its OpenFOAM trademark by Third Parties on a
licence basis. ESI Group and OpenFOAM Foundation Ltd are currently
permitted to use the Name and agreed Domain Name. For information on
trademark use, please refer to the
[trademark policy guidelines][link trademark].

Please [contact OpenCFD](http://www.openfoam.com/contact) if you have
any questions about the use of the OpenFOAM trademark.

Violations of the Trademark are monitored, and will be duly prosecuted.
