
# Conversion of Image Data from SimpleITK to Numpy and VTK

## Imports


    import numpy
    import vtk
    import SimpleITK
    from vtk.util.numpy_support import numpy_to_vtk

---

## Helper-functions

We're gonna use this function to embed a still image of a VTK render


    from IPython.display import Image
    def vtk_show(renderer, width=400, height=300):
        """
        Takes vtkRenderer instance and returns an IPython Image with the rendering.
        """
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetOffScreenRendering(1)
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(width, height)
        renderWindow.Render()
         
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        windowToImageFilter.SetInput(renderWindow)
        windowToImageFilter.Update()
         
        writer = vtk.vtkPNGWriter()
        writer.SetWriteToMemory(1)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()
        data = str(buffer(writer.GetResult()))
        
        return Image(data)

We're gonna use these functions to quickly 'convert' between SimpleITK and VTK
data types


    dctITKtoVTK = {SimpleITK.sitkInt8: vtk.VTK_TYPE_INT8,
                   SimpleITK.sitkInt16: vtk.VTK_TYPE_INT16,
                   SimpleITK.sitkInt32: vtk.VTK_TYPE_INT32,
                   SimpleITK.sitkInt64: vtk.VTK_TYPE_INT64,
                   SimpleITK.sitkUInt8: vtk.VTK_TYPE_UINT8,
                   SimpleITK.sitkUInt16: vtk.VTK_TYPE_UINT16,
                   SimpleITK.sitkUInt32: vtk.VTK_TYPE_UINT32,
                   SimpleITK.sitkUInt64: vtk.VTK_TYPE_UINT64,
                   SimpleITK.sitkFloat32: vtk.VTK_TYPE_FLOAT32,
                   SimpleITK.sitkFloat64: vtk.VTK_TYPE_FLOAT64}
    dctVTKtoITK = dict(zip(dctITKtoVTK.values(), 
                           dctITKtoVTK.keys()))
    
    def convertTypeITKtoVTK(typeITK):
        if typeITK in dctITKtoVTK:
            return dctITKtoVTK[typeITK]
        else:
            raise ValueError("Type not supported")
    
    def convertTypeVTKtoITK(typeVTK):
        if typeVTK in dctVTKtoITK:
            return dctVTKtoITK[typeVTK]
        else:
            raise ValueError("Type not supported")

---

## Options


    filenameMyHead = "./MyHead.nii"

Read in data with SimpleITK


    imgMyHead_SimpleITK = SimpleITK.ReadImage(filenameMyHead)

Convert `SimpleITK.Image` object to a `numpy.ndarray`


    imgMyHead_NumPy = numpy.ravel(SimpleITK.GetArrayFromImage(imgMyHead_SimpleITK), order='C')

Convert `numpy.ndarray` to a `vtk.vtkImageData` object


    imgMyHead_VTK = vtk.vtkImageData()
    imgMyHead_VTK.SetSpacing(imgMyHead_SimpleITK.GetSpacing())
    imgMyHead_VTK.SetOrigin(imgMyHead_SimpleITK.GetOrigin())
    imgMyHead_VTK.SetDimensions(imgMyHead_SimpleITK.GetSize())
    imgMyHead_VTK.SetScalarType(convertTypeITKtoVTK(imgMyHead_SimpleITK.GetPixelID()))
    #imgMyHead_VTK.AllocateScalars()
    imgMyHead_VTK.SetNumberOfScalarComponents(imgMyHead_SimpleITK.GetNumberOfComponentsPerPixel())


    imgMyHead_NumPyToVTK = numpy_to_vtk(imgMyHead_NumPy, 
                                        deep=True, 
                                        array_type=convertTypeITKtoVTK(imgMyHead_SimpleITK.GetPixelID()))
    
    imgMyHead_VTK.GetPointData().SetScalars(imgMyHead_NumPyToVTK)




    0




    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)
    
    origin = imgMyHead_VTK.GetOrigin()
    extent = imgMyHead_VTK.GetExtent()
    spacing = imgMyHead_VTK.GetSpacing()
    xc = origin[0] + 0.5*(extent[0] + extent[1])*spacing[0]
    yc = origin[1] + 0.5*(extent[2] + extent[3])*spacing[1]
    xd = (extent[1] - extent[0] + 1)*spacing[0]
    yd = (extent[3] - extent[2] + 1)*spacing[1]
    camera=renderer.GetActiveCamera()
    d = camera.GetDistance()
    camera.SetParallelScale(0.5*yd)
    camera.SetFocalPoint(xc,yc,0.0)
    camera.SetPosition(xc,yc,+d)
    renderer.SetActiveCamera(camera)


    mapper = vtk.vtkImageSliceMapper()
    mapper.SetInput(imgMyHead_VTK)
    mapper.SetOrientationToX()
    mapper.SetSliceNumber(imgMyHead_VTK.GetDimensions()[0] // 2)
    
    actor = vtk.vtkImageActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.5)
    
    renderer.AddActor(actor)


    vtk_show(renderer, 800, 800)




![png](ConvertSimpleITKtoVTK_files/ConvertSimpleITKtoVTK_21_0.png)




    
