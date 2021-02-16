# FrangiFilter
In this repository you can find an implementation of Frangi method for computing a probability of vessel appearance from 3D CT DICOM image.

The main steps of app execution:
- reading 3D CT image in DICOM format by ITK (Insight ToolKit) librar
- image noise is smoothed by gaussian filter from ITK
- in parallel loop (OpenMP) it initializes:
  1. a central voxel
  2. its backward, forward and central nearest neighbours (26 ones)
  3. data structures for Hessian matrix (that corresponds the central voxel),
                         eigenvector matrix and eigenvalue vector.
- solving the eigen problem
- computing vesselness function (see Frangi paper), i.e. a probability of that central voxel corresponds to a vessel
- writing an image with vesselness

Links: 
- [original paper](https://link.springer.com/chapter/10.1007/bfb0056195)
