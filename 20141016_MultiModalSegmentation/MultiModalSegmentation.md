Title: Multi-Modal Image Segmentation & Volume Rendering with Python, SimpleITK, and VTK
Author: Adamos Kyriakou
Date: Friday October 16th, 2014
Tags: Python, IPython Notebook, DICOM, VTK, ITK, SimpleITK, Medical Image Processing, Image Segmentation, Volume Rendering
Categories: Image Processing, Visualization, ITK/SimpleITK, Image Segmentation, IO, VTK

In this post I will show how to use SimpleITK to perform multi-modal segmentation on a T1 and T2 MRI dataset for better accuracy and performance. The tutorial will include  loading MHD images, uni-modal and multi-modal segmentation of the datasets, conversion of the segmented labelfields to VTK classes, and will conclude with volume-rendering techniques borrowed from VTK.

<!--more-->

---

# Introduction

## Background
In the [previous post](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/), I introduced [SimpleITK](http://www.simpleitk.org/), a simplified layer/wrapper build on top of [ITK](http://www.itk.org/), allowing for advanced image processing including but not limited to image segmentation, registration, and interpolation.

The primary strengths of SimpleITK, links and material, as well as its installation -- for vanilla and alternative Python distributions -- was extensively discussed in the last post entitled ['Image Segmentation with Python and SimpleITK'](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/). If you haven't read that post I recommend that you do so before proceeding with this one as I will not be re-visiting these topics.

Today I'll mostly be dealing with multi-modal segmentation and volume rendering.

### Multi-Modal Segmentation
A typical 

### The Dataset: The RIRE Project
[Today's dataset](https://bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/patient_109.zip) is taken from the [Retrospective Image Registration Evaluation (RIRE) Project](http://www.insight-journal.org/rire/index.php), which was *"designed to compare retrospective CT-MR and PET-MR registration techniques used by a number of groups"*. 

The [RIRE Project](http://www.insight-journal.org/rire/index.php) provides [patient datasets](http://www.insight-journal.org/rire/download_data.php) acquired with different imaging modalities, e.g., MR, CT, PET, which are meant to be used in evaluation of different image registration and segmentation techniques. The datasets are distributed in a zipped [MetaImage (`.mhd`) format](http://www.itk.org/Wiki/MetaIO/Documentation) with a [Creative Commons Attribution 3.0 United States license](http://creativecommons.org/licenses/by/3.0/us/legalcode) which pretty much [translates to](http://creativecommons.org/licenses/by/3.0/us/) "you can do whatever you want with this". Thus, these datasets are perfect for my tutorials :).


In particular, [today's dataset](https://bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/patient_109.zip) is a reduced, and slightly modified, version of the 'patient_109' dataset which can be downloaded [here](http://www.insight-journal.org/rire/download_data.php). Just [download my version](https://bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/patient_109.zip) and extract its contents alongside [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/MultiModalSegmentation.ipynb). The resulting directory structure should look something like this:

```
|____MultiModalSegmentation.ipynb
|____patient_109
| |____mr_T1
| | |____header.ascii
| | |____image.bin
| | |____patient_109_mr_T1.mhd
| |____mr_T2
| | |____header.ascii
| | |____image.bin
| | |____patient_109_mr_T2.mhd
|____patient_109.zip
```

As you can see from the above directory structure, [today's dataset](https://bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/patient_109.zip) comprises two [MetaImage (`.mhd`)](http://www.itk.org/Wiki/MetaIO/Documentation) files with a T1 and T2 MRI datasets of a single patient. In case you didn't know, the [MHD format](http://www.itk.org/Wiki/MetaIO/Documentation) is a very simple format employed heavily in the distribution of medical image data. In a nutshell its just a ASCII header, with an `.mhd` extension, which defines basic image properties, e.g., dimensions, spacing, origin, which is used to read an accompanying raw binary file, typically with a `.raw` or `.bin` extension, with the actual image data. [MHD](http://www.itk.org/Wiki/MetaIO/Documentation) files are very commonplace and supported by libraries like [VTK](http://www.vtk.org/) and [ITK](http://itk.org/), visualization software like [MayaVi](http://code.enthought.com/projects/mayavi/), [ParaView](http://www.paraview.org/), and [VisIt](https://wci.llnl.gov/simulation/computer-codes/visit), as well as image processing software like [3DSlicer](http://www.slicer.org/) and [MeVisLab](http://www.mevislab.de/).

## Summary

# Multi-Modal Segmentation

## Imports
Let's start with the imports

```
import os
import numpy
import SimpleITK
import vtk
import vtk.util.numpy_support
import matplotlib.pyplot as plt
%pylab inline
```

Once more, if you don't have a working installation of [SimpleITK](http://www.simpleitk.org/), check the [previous post](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/) where the topic is extensively discussed. In addition, if you don't happen to have [VTK](http://www.vtk.org/) then I suggest checking [this early post](http://pyscience.wordpress.com/2014/09/01/anaconda-the-creme-de-la-creme-of-python-distros-3/) and going for [Anaconda Python](https://store.continuum.io/cshop/anaconda/).

## Helper-Functions
The following 'helper-functions' are defined at the beginning of [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/MultiModalSegmentation.ipynb) and used throughout:

- `sitk_show(img, title=None, margin=0.0, dpi=40)`: This function uses `matplotlib.pyplot` to quickly visualize a 2D `SimpleITK.Image` object under the `img` parameter by first converting it to a `numpy.ndarray`. It was first introduced in [this past post about SimpleITK](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/).
- `vtk_show(renderer, width=400, height=300)`: This function allows me to pass a [`vtkRenderer`](http://www.vtk.org/doc/nightly/html/classvtkRenderer.html) object and get a PNG image output of that render, compatible with the IPython Notebook cell output. This code was presented in [this past post about VTK integration with an IPython Notebook](http://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/).

## Options
Near the beginning of [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/MultiModalSegmentation.ipynb) the we'll define a few options to keep the rest of the notebook 'clean' and allow you to make direct changes without perusing/amending the entire notebook.

```
# Paths to the .mhd files
filenameT1 = "./patient_109/mr_T1/patient_109_mr_T1.mhd"
filenameT2 = "./patient_109/mr_T2/patient_109_mr_T2.mhd"

# Slice index to visualize with 'sitk_show'
idxSlice = 26

# int label to assign to the segmented gray matter
labelGrayMatter = 1
```

As you can see the first options,  `filenameT1` and `filenameT2`, relate to the location of the accompanying `.mhd` files. Again, you need to extract the contents of [today's dataset](https://bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/patient_109.zip) next to [today's notebook](http://nbviewer.ipython.org/urls/bitbucket.org/somada141/pyscience/raw/master/20141016_MultiModalSegmentation/Material/MultiModalSegmentation.ipynb).

Unlike in the previous post, where segmentation was performed in 2D, today's post will be performing a full 3D segmentation of the entire dataset. However, as we'll be using the `sitk_show` helper-function to display the results of the segmentation, the `idxSlice` options gives us the index of the slice which we will be visualizing.

Lastly, `labelGrayMatter` is merely an integer value which will act as a label index for the gray matter in the segmentation.

## Image-Data Input
