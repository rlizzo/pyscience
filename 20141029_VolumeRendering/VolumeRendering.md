Title: Volume Rendering with Python and VTK
Author: Adamos Kyriakou
Date: Friday October 29th, 2014
Tags: Python, IPython Notebook, VTK, Medical Image Processing, Volume Rendering, Interpolation
Categories: Image Processing, Visualization, ITK/SimpleITK, Image Segmentation, IO, VTK


<!--more-->

---

# Introduction

## Background
Some of you might have read my [previous post about surface extraction](http://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/). Well in that post we performed an automatic segmentation of the bone-structures in a CT dataset and extracted a 3D surface depicting those structures. You might remember that same skull model that was used later in my [previous post about ray-casting](http://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/).

Well the 'problem' with those surface models is that they're exactly that, surfaces! Essentially they're 2D surfaces arranged in a 3D space but they're entirely hollow. Take a look at the figure below:

![3D surface model of a human skull (left), a clipped depiction through it (center), and a slice through its center (right)](figure01.png)

As you can see, what we've got here is two surfaces defining a 'pseudo-volume' but there's nothing in them which becomes obvious when we clip/slice through them. The clipping and slicing in the above figure was performed in [ParaView](http://www.paraview.org/) using the [STL model of the skull](https://bitbucket.org/somada141/pyscience/raw/master/20140910_RayCasting/Material/bones.stl) which was used in the [previous post about ray-casting](http://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/).

However, it is often the case that we want to visualize the entirety of a 3D volume, i.e., all the data that lies beneath the surface (yes, that was another one of my puns). Well, in that case we need to resort to a technique aptly termed ['volume rendering'](http://en.wikipedia.org/wiki/Volume_rendering).

Per the [VTK User's Guide](http://www.kitware.com/products/books/vtkguide.html),  "volume rendering is a term used to describe a rendering process applied to 3D data where information exists throughout a 3D space instead of simply on 2D surfaces defined in 3D space". Now, volume rendering is a inordinately popular topic in graphics and visualization. As a result, its one of the few topics in VTK that's surprisingly well documented. Due to its popularity its also one of the actively developed areas in VTK and the material I'll be presented may well be outdated a year from now (check out this [post on VTK volume rendering updates](http://www.kitware.com/source/home/post/154) on the Kitware blog).

 If you want to learn more about it I suggest you download a copy of the freely available book ['Introduction to Programming for Image Analysis with VTK'](http://medicine.yale.edu/bioimaging/suite/vtkbook/index.aspx) and read Section 12.5. Alternatively, the ['VTK User's Guide'](http://www.kitware.com/products/books/vtkguide.html) has an entire chapter dedicated to volume rendering but the book isn't as easy to come by.

### The Dataset: Brain Atlas
[Today's dataset]() comes from a project entitled ['Multi-modality MRI-based Atlas of the Brain'](http://www.spl.harvard.edu/publications/item/view/2037) by Halle et al. and it is currently available in the [Publication Database hosted by Harvard's Surgical Planning Laboratory (SPL)](https://www.spl.harvard.edu/publications/pages/display/?entriesPerPage=50&collection=1). In a nutshell, this project provides us with a very nicely segmented labelfield of the human brain with something like 150 distinguishable brain structures, along with the original medical image data.

What I did was download [this](https://www.spl.harvard.edu/publications/bitstream/download/5276) version of the atlas, which I then modified and boiled down to a compressed `.mha` file using [3DSlicer](http://www.slicer.org/). Unlike the [MHD format](http://www.itk.org/Wiki/MetaIO/Documentation), which was discussed in the [previous post about multi-modal segmentation](http://pyscience.wordpress.com/2014/11/02/multi-modal-image-segmentation-with-python-simpleitk/), this `.mha` file contains both the header and binary image data within the same file. In addition, I modified the accompanying color-file, which is essentially a [CSV file](http://en.wikipedia.org/wiki/Comma-separated_values) listing every index in the labelfield along with the name of the represented brain structure and a suggested RGB color.

What you need to do is download [today's dataset](), and extract the contents of the `.zip` file alongside [today's notebook]().

## Summary

---

# Volume Rendering with Python and VTK

## Imports
As always, we'll be starting with the imports:

```
import os
import numpy
import vtk
```

## Helper-Functions
The following 'helper-functions' are defined at the beginning of [today's notebook]() and used throughout:


## Options
Near the beginning of [today's notebook]() we'll define a few options to keep the rest of the notebook 'clean' and allow you to make direct changes without perusing/amending the entire notebook.

- `vtk_show(renderer, width=400, height=300)`: This function allows me to pass a [`vtkRenderer`](http://www.vtk.org/doc/nightly/html/classvtkRenderer.html) object and get a PNG image output of that render, compatible with the IPython Notebook cell output. This code was presented in [this past post about VTK integration with an IPython Notebook](http://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/).
- 
- `l2n = lambda l: numpy.array(l)` and `n2l = lambda n: list(n)`: Two simple `lambda` functions meant to quickly convert a `list` or `tuple` to a `numpy.ndarray` and vice-versa. These function were first used in [this past post about ray-tracing with VTK](http://pyscience.wordpress.com/2014/10/05/from-ray-casting-to-ray-tracing-with-python-and-vtk/).



# Volume Rendering
Per the [VTK User's Guide](http://www.kitware.com/products/books/vtkguide.html),  "volume rendering is a term used to describe a rendering process applied to 3D data where information exists throughout a 3D space instead of simply on 2D surfaces defined in 3D space". You might remember the [surface extraction post](http://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/) where we rendered a 3D skull extracted from segmented medical image data. Well the title says it all: surface extraction. What we did back then was create a closed 2D surface defined in 3D space but it was just that, a surface! The result of volume rendering isn't a hollow surface but rather a full 3D volume. 

Volume rendering in VTK is performed through the [`vtkVolumeRayCastMapper`](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastMapper.html) class, which, to my knowledge, only works with  [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) objects.

## Convert [`SimpleITK.Image`](http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html) to [`vtk.vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html)
Firstly, we need to convert `imgGrayMatterComp`, the [`SimpleITK.Image`](http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html) object containing the results of our multi-modal segmentation to [`vtk.vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) in order to perform the volume rendering. For that we need to first convert the [`SimpleITK.Image`](http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html) object to a [`numpy.ndarray`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) and then convert that to a [`vtk.vtkObject`](http://www.vtk.org/doc/nightly/html/classvtkObject.html). Finally we'll create a [`vtk.vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) object with the appropriate image properties and the converted image data.

Let's see the code:

```
# Create a new 'vtkImageData' object and set basic properties
vtkGrayMatter = vtk.vtkImageData()
vtkGrayMatter.SetSpacing(imgGrayMatterComp.GetSpacing())
vtkGrayMatter.SetOrigin(imgGrayMatterComp.GetOrigin())
vtkGrayMatter.SetDimensions(imgGrayMatterComp.GetSize())
vtkGrayMatter.SetNumberOfScalarComponents(imgGrayMatterComp.GetNumberOfComponentsPerPixel())

# Convert the 'SimpleITK.Image' to 'numpy.ndarray'
imgGrayMatterComp_numpy = numpy.ravel(SimpleITK.GetArrayFromImage(imgGrayMatterComp),
                                      order='C')

# Convert the 'numpy.ndarray' to a 'vtkObject'
imgGrayMatterComp_vtk = numpy_to_vtk(imgGrayMatterComp_numpy, 
                                     deep=True, 
                                     array_type=vtk.VTK_UNSIGNED_SHORT)

vtkGrayMatter.GetPointData().SetScalars(imgGrayMatterComp_vtk)
```

As you can see we start by creating a new [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) object under `vtkGrayMatter`. We then set the basic image properties, i.e., spacing, origin, dimensions, number of components. What's interesting is that we can do so directly by using the [`SimpleITK.Image`](http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html) class methods which return tuples of the image properties and which can be directly used as input for the corresponding [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) methods.  At this point the [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html)  named `vtkGrayMatter` has been properly initialized but contains no image data. 

Using the `SimpleITK.GetArrayFromImage` function we convert the results of the multi-modal segmentation residing under `imgGrayMatterComp` to a `numpy.ndarray` under the name of `imgGrayMatterComp_numpy`. Note that we use [`numpy.ravel`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html) to create a 1D, C-ordered representation of the 3D image data. The conversion of a [`SimpleITK.Image`](http://www.itk.org/SimpleITKDoxygen/html/classitk_1_1simple_1_1Image.html) object to a [`numpy.ndarray`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) was shown in this [past post about SimpleITK](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/).

Next, we convert `imgGrayMatterComp_numpy` to a format that can be understood by VTK. We use the `numpy_to_vtk` function, residing under the `vtk.util.numpy_support` module, to create a `vtkObject` under the name of `imgGrayMatterComp_vtk`. Pay attention to the fact that we're deep-copying the contents of `imgGrayMatterComp_numpy` to avoid issues with memory deallocation and that we're manually setting the type of the array to `vtk.VTK_UNSIGNED_SHORT`. The conversion of a [`numpy.ndarray`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) to [`vtk.vtkObject`](http://www.vtk.org/doc/nightly/html/classvtkObject.html) was shown in [this previous post](http://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/).

Finally, through the use of  `GetPointData().SetScalars()` we set the image data scalars of `vtkGrayMatter` to the data contained in `imgGrayMatterComp_vtk`. Now, `vtkGrayMatter` is a fully-fledged [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) object.

> Note that the conversion seen above is an extremely useful process when one wants to manipulate `numpy.ndarray` data in VTK. An enormous amount of VTK functionality is applicable to [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) objects and I suggest you become familiar with the above code.

### Prepare volume-renderer
As mentioned before, volume rendering in VTK is performed through the [`vtkVolumeRayCastMapper`](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastMapper.html) class. Now that we have our data in a  [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html) object under the name of `vtkGrayMatter` we're ready to proceed.

As the name [`vtkVolumeRayCastMapper`](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastMapper.html) implies, volume rendering in VTK is performed through ray-casting operations on the [`vtkImageData`](http://www.vtk.org/doc/nightly/html/classvtkImageData.html). As a first step we need to choose and define the ['vtkVolumeRayCastFunction'](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastFunction.html) we'll be using with [`vtkVolumeRayCastMapper`](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastMapper.html). 

Currently in VTK, three ['vtkVolumeRayCastFunction'](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastFunction.html) classes have been implemented:

- [`vtkVolumeRayCastCompositeFunction`](http://www.vtk.org/doc/nightly/html/classvtkVolumeRayCastCompositeFunction.html)

```
funcRayCast = vtk.vtkVolumeRayCastCompositeFunction()
funcRayCast.SetCompositeMethodToInterpolateFirst()

mapperVolume = vtk.vtkVolumeRayCastMapper()
mapperVolume.SetVolumeRayCastFunction(funcRayCast)
mapperVolume.SetInput(vtkGrayMatter)
```
## Links & Resources

### Material
Here's the material used in this post:

- [IPython Notebook]() with the entire process.
- [Modified Brain Atlas Dataset]() used in this post.

### See also

Check out these past posts which were used and referenced today or are relevant to this post:

- [Anaconda: The crème de la crème of Python distros](http://pyscience.wordpress.com/2014/09/01/anaconda-the-creme-de-la-creme-of-python-distros-3/)
- [IPython Notebook & VTK](http://pyscience.wordpress.com/2014/09/03/ipython-notebook-vtk/)
- [NumPy to VTK: Converting your NumPy arrays to VTK arrays and files](http://pyscience.wordpress.com/2014/09/06/numpy-to-vtk-converting-your-numpy-arrays-to-vtk-arrays-and-files/)
- [DICOM in Python: Importing medical image data into NumPy with PyDICOM and VTK](http://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/)
- [Surface Extraction: Creating a mesh from pixel-data using Python and VTK](http://pyscience.wordpress.com/2014/09/11/surface-extraction-creating-a-mesh-from-pixel-data-using-python-and-vtk/)
- [Ray Casting with Python and VTK: Intersecting lines/rays with surface meshes](http://pyscience.wordpress.com/2014/09/21/ray-casting-with-python-and-vtk-intersecting-linesrays-with-surface-meshes/)
- [Image Segmentation with Python and SimpleITK](http://pyscience.wordpress.com/2014/10/19/image-segmentation-with-python-and-simpleitk/)
- [Multi-Modal Image Segmentation with Python & SimpleITK](http://pyscience.wordpress.com/2014/11/02/multi-modal-image-segmentation-with-python-simpleitk/)

> Don't forget: all material I'm presenting in this blog can be found under the [PyScience BitBucket repository](https://bitbucket.org/somada141/pyscience).
