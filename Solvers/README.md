# Description of the different solvers:
  - ***myPimpleADDCFinal***: Pimple solver employing the Automatic Differentaible (AD) version of OpenFOAM. This solver accepts the field Ufaces as input and incomporates it to the momentum equation using a Deffered Correction (DC) approach. The solver is compiled to communicate with Python through our RAM communication technique for URANS simulations.
  - ***myPimpleADDCFinalLES***: Same implementation as the previous one, but for LES simulations.
  - ***myPimpleVJPDCFinal***: Reverse solver of myPimpleADDCFinal. Returns the derivative dU/dUfaces employing Automatic Differentiation.
  - ***myPimpleVJPDCFinalLES***: Reverse solver of myPimpleADDCFinalLES. Returns the derivative dU/dUfaces employing Automatic Differentiation.
  - ***pimpleFOAMSMfinal***: Classic URANS Pimple solver modified to accept the RAM communication from Python.
  - ***pimpleFOAMSMfinalLES***: Classic LES Pimple solver modified to accept the RAM communication from Python.

# How to compile the solvers?:
The solvers may be compiled using make following the commands:
```
wclean
wmake
```
