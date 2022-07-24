import sys

# trace generated using paraview version 5.8.0
#
# To ensure correct image size when batch processing, please search
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *

#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'
aafoam = OpenFOAMReader(FileName="a.foam")
aafoam.SkipZeroTime = 1
aafoam.CaseType = "Reconstructed Case"
aafoam.LabelSize = "32-bit"
aafoam.ScalarSize = "64-bit (DP)"
aafoam.Createcelltopointfiltereddata = 1
aafoam.Adddimensionalunitstoarraynames = 0
aafoam.MeshRegions = ["internalMesh"]
aafoam.CellArrays = ["U", "p", "s", "vorticityField"]
aafoam.PointArrays = []
aafoam.LagrangianArrays = []
aafoam.Cachemesh = 1
aafoam.Decomposepolyhedra = 1
aafoam.ListtimestepsaccordingtocontrolDict = 0
aafoam.Lagrangianpositionswithoutextradata = 1
aafoam.Readzones = 0
aafoam.Copydatatocellzones = 0

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate("RenderView")
# uncomment following to set a specific view size
# renderView1.ViewSize = [1165, 803]

# get layout
layout1 = GetLayout()

# show data in view
aafoamDisplay = Show(aafoam, renderView1, "UnstructuredGridRepresentation")

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction("p")
pLUT.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
pLUT.InterpretValuesAsCategories = 0
pLUT.AnnotationsInitialized = 0
pLUT.ShowCategoricalColorsinDataRangeOnly = 0
pLUT.RescaleOnVisibilityChange = 0
pLUT.EnableOpacityMapping = 0
pLUT.RGBPoints = [
    -0.34813499450683594,
    0.231373,
    0.298039,
    0.752941,
    -0.17406749725341797,
    0.865003,
    0.865003,
    0.865003,
    0.0,
    0.705882,
    0.0156863,
    0.14902,
]
pLUT.UseLogScale = 0
pLUT.ShowDataHistogram = 0
pLUT.AutomaticDataHistogramComputation = 0
pLUT.DataHistogramNumberOfBins = 10
pLUT.ColorSpace = "Diverging"
pLUT.UseBelowRangeColor = 0
pLUT.BelowRangeColor = [0.0, 0.0, 0.0]
pLUT.UseAboveRangeColor = 0
pLUT.AboveRangeColor = [0.5, 0.5, 0.5]
pLUT.NanColor = [1.0, 1.0, 0.0]
pLUT.NanOpacity = 1.0
pLUT.Discretize = 1
pLUT.NumberOfTableValues = 256
pLUT.ScalarRangeInitialized = 1.0
pLUT.HSVWrap = 0
pLUT.VectorComponent = 0
pLUT.VectorMode = "Magnitude"
pLUT.AllowDuplicateScalars = 1
pLUT.Annotations = []
pLUT.ActiveAnnotatedValues = []
pLUT.IndexedColors = []
pLUT.IndexedOpacities = []

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction("p")
pPWF.Points = [-0.34813499450683594, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]
pPWF.AllowDuplicateScalars = 1
pPWF.UseLogScale = 0
pPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
aafoamDisplay.Representation = "Surface"
aafoamDisplay.ColorArrayName = ["POINTS", "p"]
aafoamDisplay.LookupTable = pLUT
aafoamDisplay.MapScalars = 1
aafoamDisplay.MultiComponentsMapping = 0
aafoamDisplay.InterpolateScalarsBeforeMapping = 1
aafoamDisplay.Opacity = 1.0
aafoamDisplay.PointSize = 2.0
aafoamDisplay.LineWidth = 1.0
aafoamDisplay.RenderLinesAsTubes = 0
aafoamDisplay.RenderPointsAsSpheres = 0
aafoamDisplay.Interpolation = "Gouraud"
aafoamDisplay.Specular = 0.0
aafoamDisplay.SpecularColor = [1.0, 1.0, 1.0]
aafoamDisplay.SpecularPower = 100.0
aafoamDisplay.Luminosity = 0.0
aafoamDisplay.Ambient = 0.0
aafoamDisplay.Diffuse = 1.0
aafoamDisplay.Roughness = 0.3
aafoamDisplay.Metallic = 0.0
aafoamDisplay.Texture = None
aafoamDisplay.RepeatTextures = 1
aafoamDisplay.InterpolateTextures = 0
aafoamDisplay.SeamlessU = 0
aafoamDisplay.SeamlessV = 0
aafoamDisplay.UseMipmapTextures = 0
aafoamDisplay.BaseColorTexture = None
aafoamDisplay.NormalTexture = None
aafoamDisplay.NormalScale = 1.0
aafoamDisplay.MaterialTexture = None
aafoamDisplay.OcclusionStrength = 1.0
aafoamDisplay.EmissiveTexture = None
aafoamDisplay.EmissiveFactor = [1.0, 1.0, 1.0]
aafoamDisplay.FlipTextures = 0
aafoamDisplay.BackfaceRepresentation = "Follow Frontface"
aafoamDisplay.BackfaceAmbientColor = [1.0, 1.0, 1.0]
aafoamDisplay.BackfaceOpacity = 1.0
aafoamDisplay.Position = [0.0, 0.0, 0.0]
aafoamDisplay.Scale = [1.0, 1.0, 1.0]
aafoamDisplay.Orientation = [0.0, 0.0, 0.0]
aafoamDisplay.Origin = [0.0, 0.0, 0.0]
aafoamDisplay.Pickable = 1
aafoamDisplay.Triangulate = 0
aafoamDisplay.UseShaderReplacements = 0
aafoamDisplay.ShaderReplacements = ""
aafoamDisplay.NonlinearSubdivisionLevel = 1
aafoamDisplay.UseDataPartitions = 0
aafoamDisplay.OSPRayUseScaleArray = 0
aafoamDisplay.OSPRayScaleArray = "p"
aafoamDisplay.OSPRayScaleFunction = "PiecewiseFunction"
aafoamDisplay.OSPRayMaterial = "None"
aafoamDisplay.Orient = 0
aafoamDisplay.OrientationMode = "Direction"
aafoamDisplay.SelectOrientationVectors = "U"
aafoamDisplay.Scaling = 0
aafoamDisplay.ScaleMode = "No Data Scaling Off"
aafoamDisplay.ScaleFactor = 0.002999999932944775
aafoamDisplay.SelectScaleArray = "p"
aafoamDisplay.GlyphType = "Arrow"
aafoamDisplay.UseGlyphTable = 0
aafoamDisplay.GlyphTableIndexArray = "p"
aafoamDisplay.UseCompositeGlyphTable = 0
aafoamDisplay.UseGlyphCullingAndLOD = 0
aafoamDisplay.LODValues = []
aafoamDisplay.ColorByLODIndex = 0
aafoamDisplay.GaussianRadius = 0.00014999999664723872
aafoamDisplay.ShaderPreset = "Sphere"
aafoamDisplay.CustomTriangleScale = 3
aafoamDisplay.CustomShader = """ // This custom shader code define a gaussian blur
// Please take a look into vtkSMPointGaussianRepresentation.cxx
// for other custom shader examples
//VTK::Color::Impl
  float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
  float gaussian = exp(-0.5*dist2);
  opacity = opacity*gaussian;
"""
aafoamDisplay.Emissive = 0
aafoamDisplay.ScaleByArray = 0
aafoamDisplay.SetScaleArray = ["POINTS", "p"]
aafoamDisplay.ScaleArrayComponent = ""
aafoamDisplay.UseScaleFunction = 1
aafoamDisplay.ScaleTransferFunction = "PiecewiseFunction"
aafoamDisplay.OpacityByArray = 0
aafoamDisplay.OpacityArray = ["POINTS", "p"]
aafoamDisplay.OpacityArrayComponent = ""
aafoamDisplay.OpacityTransferFunction = "PiecewiseFunction"
aafoamDisplay.DataAxesGrid = "GridAxesRepresentation"
aafoamDisplay.SelectionCellLabelBold = 0
aafoamDisplay.SelectionCellLabelColor = [0.0, 1.0, 0.0]
aafoamDisplay.SelectionCellLabelFontFamily = "Arial"
aafoamDisplay.SelectionCellLabelFontFile = ""
aafoamDisplay.SelectionCellLabelFontSize = 18
aafoamDisplay.SelectionCellLabelItalic = 0
aafoamDisplay.SelectionCellLabelJustification = "Left"
aafoamDisplay.SelectionCellLabelOpacity = 1.0
aafoamDisplay.SelectionCellLabelShadow = 0
aafoamDisplay.SelectionPointLabelBold = 0
aafoamDisplay.SelectionPointLabelColor = [1.0, 1.0, 0.0]
aafoamDisplay.SelectionPointLabelFontFamily = "Arial"
aafoamDisplay.SelectionPointLabelFontFile = ""
aafoamDisplay.SelectionPointLabelFontSize = 18
aafoamDisplay.SelectionPointLabelItalic = 0
aafoamDisplay.SelectionPointLabelJustification = "Left"
aafoamDisplay.SelectionPointLabelOpacity = 1.0
aafoamDisplay.SelectionPointLabelShadow = 0
aafoamDisplay.PolarAxes = "PolarAxesRepresentation"
aafoamDisplay.ScalarOpacityFunction = pPWF
aafoamDisplay.ScalarOpacityUnitDistance = 0.0006518277049957368
# aafoamDisplay.ExtractedBlockIndex = 1
aafoamDisplay.SelectMapper = "Projected tetra"
aafoamDisplay.SamplingDimensions = [128, 128, 128]
aafoamDisplay.UseFloatingPointFrameBuffer = 1

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
aafoamDisplay.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
aafoamDisplay.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
aafoamDisplay.GlyphType.TipResolution = 6
aafoamDisplay.GlyphType.TipRadius = 0.1
aafoamDisplay.GlyphType.TipLength = 0.35
aafoamDisplay.GlyphType.ShaftResolution = 6
aafoamDisplay.GlyphType.ShaftRadius = 0.03
aafoamDisplay.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
aafoamDisplay.ScaleTransferFunction.Points = [
    -0.34813499450683594,
    0.0,
    0.5,
    0.0,
    0.0,
    1.0,
    0.5,
    0.0,
]
aafoamDisplay.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
aafoamDisplay.OpacityTransferFunction.Points = [
    -0.34813499450683594,
    0.0,
    0.5,
    0.0,
    0.0,
    1.0,
    0.5,
    0.0,
]
aafoamDisplay.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
aafoamDisplay.DataAxesGrid.XTitle = "X Axis"
aafoamDisplay.DataAxesGrid.YTitle = "Y Axis"
aafoamDisplay.DataAxesGrid.ZTitle = "Z Axis"
aafoamDisplay.DataAxesGrid.XTitleFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.XTitleFontFile = ""
aafoamDisplay.DataAxesGrid.XTitleBold = 0
aafoamDisplay.DataAxesGrid.XTitleItalic = 0
aafoamDisplay.DataAxesGrid.XTitleFontSize = 12
aafoamDisplay.DataAxesGrid.XTitleShadow = 0
aafoamDisplay.DataAxesGrid.XTitleOpacity = 1.0
aafoamDisplay.DataAxesGrid.YTitleFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.YTitleFontFile = ""
aafoamDisplay.DataAxesGrid.YTitleBold = 0
aafoamDisplay.DataAxesGrid.YTitleItalic = 0
aafoamDisplay.DataAxesGrid.YTitleFontSize = 12
aafoamDisplay.DataAxesGrid.YTitleShadow = 0
aafoamDisplay.DataAxesGrid.YTitleOpacity = 1.0
aafoamDisplay.DataAxesGrid.ZTitleFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.ZTitleFontFile = ""
aafoamDisplay.DataAxesGrid.ZTitleBold = 0
aafoamDisplay.DataAxesGrid.ZTitleItalic = 0
aafoamDisplay.DataAxesGrid.ZTitleFontSize = 12
aafoamDisplay.DataAxesGrid.ZTitleShadow = 0
aafoamDisplay.DataAxesGrid.ZTitleOpacity = 1.0
aafoamDisplay.DataAxesGrid.FacesToRender = 63
aafoamDisplay.DataAxesGrid.CullBackface = 0
aafoamDisplay.DataAxesGrid.CullFrontface = 1
aafoamDisplay.DataAxesGrid.ShowGrid = 0
aafoamDisplay.DataAxesGrid.ShowEdges = 1
aafoamDisplay.DataAxesGrid.ShowTicks = 1
aafoamDisplay.DataAxesGrid.LabelUniqueEdgesOnly = 1
aafoamDisplay.DataAxesGrid.AxesToLabel = 63
aafoamDisplay.DataAxesGrid.XLabelFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.XLabelFontFile = ""
aafoamDisplay.DataAxesGrid.XLabelBold = 0
aafoamDisplay.DataAxesGrid.XLabelItalic = 0
aafoamDisplay.DataAxesGrid.XLabelFontSize = 12
aafoamDisplay.DataAxesGrid.XLabelShadow = 0
aafoamDisplay.DataAxesGrid.XLabelOpacity = 1.0
aafoamDisplay.DataAxesGrid.YLabelFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.YLabelFontFile = ""
aafoamDisplay.DataAxesGrid.YLabelBold = 0
aafoamDisplay.DataAxesGrid.YLabelItalic = 0
aafoamDisplay.DataAxesGrid.YLabelFontSize = 12
aafoamDisplay.DataAxesGrid.YLabelShadow = 0
aafoamDisplay.DataAxesGrid.YLabelOpacity = 1.0
aafoamDisplay.DataAxesGrid.ZLabelFontFamily = "Arial"
aafoamDisplay.DataAxesGrid.ZLabelFontFile = ""
aafoamDisplay.DataAxesGrid.ZLabelBold = 0
aafoamDisplay.DataAxesGrid.ZLabelItalic = 0
aafoamDisplay.DataAxesGrid.ZLabelFontSize = 12
aafoamDisplay.DataAxesGrid.ZLabelShadow = 0
aafoamDisplay.DataAxesGrid.ZLabelOpacity = 1.0
aafoamDisplay.DataAxesGrid.XAxisNotation = "Mixed"
aafoamDisplay.DataAxesGrid.XAxisPrecision = 2
aafoamDisplay.DataAxesGrid.XAxisUseCustomLabels = 0
aafoamDisplay.DataAxesGrid.XAxisLabels = []
aafoamDisplay.DataAxesGrid.YAxisNotation = "Mixed"
aafoamDisplay.DataAxesGrid.YAxisPrecision = 2
aafoamDisplay.DataAxesGrid.YAxisUseCustomLabels = 0
aafoamDisplay.DataAxesGrid.YAxisLabels = []
aafoamDisplay.DataAxesGrid.ZAxisNotation = "Mixed"
aafoamDisplay.DataAxesGrid.ZAxisPrecision = 2
aafoamDisplay.DataAxesGrid.ZAxisUseCustomLabels = 0
aafoamDisplay.DataAxesGrid.ZAxisLabels = []
aafoamDisplay.DataAxesGrid.UseCustomBounds = 0
aafoamDisplay.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
aafoamDisplay.PolarAxes.Visibility = 0
aafoamDisplay.PolarAxes.Translation = [0.0, 0.0, 0.0]
aafoamDisplay.PolarAxes.Scale = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.Orientation = [0.0, 0.0, 0.0]
aafoamDisplay.PolarAxes.EnableCustomBounds = [0, 0, 0]
aafoamDisplay.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
aafoamDisplay.PolarAxes.EnableCustomRange = 0
aafoamDisplay.PolarAxes.CustomRange = [0.0, 1.0]
aafoamDisplay.PolarAxes.PolarAxisVisibility = 1
aafoamDisplay.PolarAxes.RadialAxesVisibility = 1
aafoamDisplay.PolarAxes.DrawRadialGridlines = 1
aafoamDisplay.PolarAxes.PolarArcsVisibility = 1
aafoamDisplay.PolarAxes.DrawPolarArcsGridlines = 1
aafoamDisplay.PolarAxes.NumberOfRadialAxes = 0
aafoamDisplay.PolarAxes.AutoSubdividePolarAxis = 1
aafoamDisplay.PolarAxes.NumberOfPolarAxis = 0
aafoamDisplay.PolarAxes.MinimumRadius = 0.0
aafoamDisplay.PolarAxes.MinimumAngle = 0.0
aafoamDisplay.PolarAxes.MaximumAngle = 90.0
aafoamDisplay.PolarAxes.RadialAxesOriginToPolarAxis = 1
aafoamDisplay.PolarAxes.Ratio = 1.0
aafoamDisplay.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
aafoamDisplay.PolarAxes.PolarAxisTitleVisibility = 1
aafoamDisplay.PolarAxes.PolarAxisTitle = "Radial Distance"
aafoamDisplay.PolarAxes.PolarAxisTitleLocation = "Bottom"
aafoamDisplay.PolarAxes.PolarLabelVisibility = 1
aafoamDisplay.PolarAxes.PolarLabelFormat = "%-#6.3g"
aafoamDisplay.PolarAxes.PolarLabelExponentLocation = "Labels"
aafoamDisplay.PolarAxes.RadialLabelVisibility = 1
aafoamDisplay.PolarAxes.RadialLabelFormat = "%-#3.1f"
aafoamDisplay.PolarAxes.RadialLabelLocation = "Bottom"
aafoamDisplay.PolarAxes.RadialUnitsVisibility = 1
aafoamDisplay.PolarAxes.ScreenSize = 10.0
aafoamDisplay.PolarAxes.PolarAxisTitleOpacity = 1.0
aafoamDisplay.PolarAxes.PolarAxisTitleFontFamily = "Arial"
aafoamDisplay.PolarAxes.PolarAxisTitleFontFile = ""
aafoamDisplay.PolarAxes.PolarAxisTitleBold = 0
aafoamDisplay.PolarAxes.PolarAxisTitleItalic = 0
aafoamDisplay.PolarAxes.PolarAxisTitleShadow = 0
aafoamDisplay.PolarAxes.PolarAxisTitleFontSize = 12
aafoamDisplay.PolarAxes.PolarAxisLabelOpacity = 1.0
aafoamDisplay.PolarAxes.PolarAxisLabelFontFamily = "Arial"
aafoamDisplay.PolarAxes.PolarAxisLabelFontFile = ""
aafoamDisplay.PolarAxes.PolarAxisLabelBold = 0
aafoamDisplay.PolarAxes.PolarAxisLabelItalic = 0
aafoamDisplay.PolarAxes.PolarAxisLabelShadow = 0
aafoamDisplay.PolarAxes.PolarAxisLabelFontSize = 12
aafoamDisplay.PolarAxes.LastRadialAxisTextOpacity = 1.0
aafoamDisplay.PolarAxes.LastRadialAxisTextFontFamily = "Arial"
aafoamDisplay.PolarAxes.LastRadialAxisTextFontFile = ""
aafoamDisplay.PolarAxes.LastRadialAxisTextBold = 0
aafoamDisplay.PolarAxes.LastRadialAxisTextItalic = 0
aafoamDisplay.PolarAxes.LastRadialAxisTextShadow = 0
aafoamDisplay.PolarAxes.LastRadialAxisTextFontSize = 12
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextFontFamily = "Arial"
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ""
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextBold = 0
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextItalic = 0
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextShadow = 0
aafoamDisplay.PolarAxes.SecondaryRadialAxesTextFontSize = 12
aafoamDisplay.PolarAxes.EnableDistanceLOD = 1
aafoamDisplay.PolarAxes.DistanceLODThreshold = 0.7
aafoamDisplay.PolarAxes.EnableViewAngleLOD = 1
aafoamDisplay.PolarAxes.ViewAngleLODThreshold = 0.7
aafoamDisplay.PolarAxes.SmallestVisiblePolarAngle = 0.5
aafoamDisplay.PolarAxes.PolarTicksVisibility = 1
aafoamDisplay.PolarAxes.ArcTicksOriginToPolarAxis = 1
aafoamDisplay.PolarAxes.TickLocation = "Both"
aafoamDisplay.PolarAxes.AxisTickVisibility = 1
aafoamDisplay.PolarAxes.AxisMinorTickVisibility = 0
aafoamDisplay.PolarAxes.ArcTickVisibility = 1
aafoamDisplay.PolarAxes.ArcMinorTickVisibility = 0
aafoamDisplay.PolarAxes.DeltaAngleMajor = 10.0
aafoamDisplay.PolarAxes.DeltaAngleMinor = 5.0
aafoamDisplay.PolarAxes.PolarAxisMajorTickSize = 0.0
aafoamDisplay.PolarAxes.PolarAxisTickRatioSize = 0.3
aafoamDisplay.PolarAxes.PolarAxisMajorTickThickness = 1.0
aafoamDisplay.PolarAxes.PolarAxisTickRatioThickness = 0.5
aafoamDisplay.PolarAxes.LastRadialAxisMajorTickSize = 0.0
aafoamDisplay.PolarAxes.LastRadialAxisTickRatioSize = 0.3
aafoamDisplay.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
aafoamDisplay.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
aafoamDisplay.PolarAxes.ArcMajorTickSize = 0.0
aafoamDisplay.PolarAxes.ArcTickRatioSize = 0.3
aafoamDisplay.PolarAxes.ArcMajorTickThickness = 1.0
aafoamDisplay.PolarAxes.ArcTickRatioThickness = 0.5
aafoamDisplay.PolarAxes.Use2DMode = 0
aafoamDisplay.PolarAxes.UseLogAxis = 0

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
aafoamDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Slice'
slice1 = Slice(Input=aafoam)
slice1.SliceType = "Plane"
slice1.HyperTreeGridSlicer = "Plane"
slice1.UseDual = 0
slice1.Crinkleslice = 0
slice1.Triangulatetheslice = 1
slice1.Mergeduplicatedpointsintheslice = 1
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.0, -3.497116267681122e-07, 0.004999985103495419]
slice1.SliceType.Normal = [1.0, 0.0, 0.0]
slice1.SliceType.Offset = 0.0

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.0, -3.497116267681122e-07, 0.004999985103495419]
slice1.HyperTreeGridSlicer.Normal = [1.0, 0.0, 0.0]
slice1.HyperTreeGridSlicer.Offset = 0.0

# show data in view
slice1Display = Show(slice1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
slice1Display.Representation = "Surface"
slice1Display.ColorArrayName = ["POINTS", "p"]
slice1Display.LookupTable = pLUT
slice1Display.MapScalars = 1
slice1Display.MultiComponentsMapping = 0
slice1Display.InterpolateScalarsBeforeMapping = 1
slice1Display.Opacity = 1.0
slice1Display.PointSize = 2.0
slice1Display.LineWidth = 1.0
slice1Display.RenderLinesAsTubes = 0
slice1Display.RenderPointsAsSpheres = 0
slice1Display.Interpolation = "Gouraud"
slice1Display.Specular = 0.0
slice1Display.SpecularColor = [1.0, 1.0, 1.0]
slice1Display.SpecularPower = 100.0
slice1Display.Luminosity = 0.0
slice1Display.Ambient = 0.0
slice1Display.Diffuse = 1.0
slice1Display.Roughness = 0.3
slice1Display.Metallic = 0.0
slice1Display.Texture = None
slice1Display.RepeatTextures = 1
slice1Display.InterpolateTextures = 0
slice1Display.SeamlessU = 0
slice1Display.SeamlessV = 0
slice1Display.UseMipmapTextures = 0
slice1Display.BaseColorTexture = None
slice1Display.NormalTexture = None
slice1Display.NormalScale = 1.0
slice1Display.MaterialTexture = None
slice1Display.OcclusionStrength = 1.0
slice1Display.EmissiveTexture = None
slice1Display.EmissiveFactor = [1.0, 1.0, 1.0]
slice1Display.FlipTextures = 0
slice1Display.BackfaceRepresentation = "Follow Frontface"
slice1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
slice1Display.BackfaceOpacity = 1.0
slice1Display.Position = [0.0, 0.0, 0.0]
slice1Display.Scale = [1.0, 1.0, 1.0]
slice1Display.Orientation = [0.0, 0.0, 0.0]
slice1Display.Origin = [0.0, 0.0, 0.0]
slice1Display.Pickable = 1
slice1Display.Triangulate = 0
slice1Display.UseShaderReplacements = 0
slice1Display.ShaderReplacements = ""
slice1Display.NonlinearSubdivisionLevel = 1
slice1Display.UseDataPartitions = 0
slice1Display.OSPRayUseScaleArray = 0
slice1Display.OSPRayScaleArray = "p"
slice1Display.OSPRayScaleFunction = "PiecewiseFunction"
slice1Display.OSPRayMaterial = "None"
slice1Display.Orient = 0
slice1Display.OrientationMode = "Direction"
slice1Display.SelectOrientationVectors = "U"
slice1Display.Scaling = 0
slice1Display.ScaleMode = "No Data Scaling Off"
slice1Display.ScaleFactor = 0.002999929804354906
slice1Display.SelectScaleArray = "p"
slice1Display.GlyphType = "Arrow"
slice1Display.UseGlyphTable = 0
slice1Display.GlyphTableIndexArray = "p"
slice1Display.UseCompositeGlyphTable = 0
slice1Display.UseGlyphCullingAndLOD = 0
slice1Display.LODValues = []
slice1Display.ColorByLODIndex = 0
slice1Display.GaussianRadius = 0.0001499964902177453
slice1Display.ShaderPreset = "Sphere"
slice1Display.CustomTriangleScale = 3
slice1Display.CustomShader = """ // This custom shader code define a gaussian blur
// Please take a look into vtkSMPointGaussianRepresentation.cxx
// for other custom shader examples
//VTK::Color::Impl
  float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
  float gaussian = exp(-0.5*dist2);
  opacity = opacity*gaussian;
"""
slice1Display.Emissive = 0
slice1Display.ScaleByArray = 0
slice1Display.SetScaleArray = ["POINTS", "p"]
slice1Display.ScaleArrayComponent = ""
slice1Display.UseScaleFunction = 1
slice1Display.ScaleTransferFunction = "PiecewiseFunction"
slice1Display.OpacityByArray = 0
slice1Display.OpacityArray = ["POINTS", "p"]
slice1Display.OpacityArrayComponent = ""
slice1Display.OpacityTransferFunction = "PiecewiseFunction"
slice1Display.DataAxesGrid = "GridAxesRepresentation"
slice1Display.SelectionCellLabelBold = 0
slice1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
slice1Display.SelectionCellLabelFontFamily = "Arial"
slice1Display.SelectionCellLabelFontFile = ""
slice1Display.SelectionCellLabelFontSize = 18
slice1Display.SelectionCellLabelItalic = 0
slice1Display.SelectionCellLabelJustification = "Left"
slice1Display.SelectionCellLabelOpacity = 1.0
slice1Display.SelectionCellLabelShadow = 0
slice1Display.SelectionPointLabelBold = 0
slice1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
slice1Display.SelectionPointLabelFontFamily = "Arial"
slice1Display.SelectionPointLabelFontFile = ""
slice1Display.SelectionPointLabelFontSize = 18
slice1Display.SelectionPointLabelItalic = 0
slice1Display.SelectionPointLabelJustification = "Left"
slice1Display.SelectionPointLabelOpacity = 1.0
slice1Display.SelectionPointLabelShadow = 0
slice1Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
slice1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
slice1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
slice1Display.GlyphType.TipResolution = 6
slice1Display.GlyphType.TipRadius = 0.1
slice1Display.GlyphType.TipLength = 0.35
slice1Display.GlyphType.ShaftResolution = 6
slice1Display.GlyphType.ShaftRadius = 0.03
slice1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [
    -0.30105113983154297,
    0.0,
    0.5,
    0.0,
    -0.0472472608089447,
    1.0,
    0.5,
    0.0,
]
slice1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [
    -0.30105113983154297,
    0.0,
    0.5,
    0.0,
    -0.0472472608089447,
    1.0,
    0.5,
    0.0,
]
slice1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
slice1Display.DataAxesGrid.XTitle = "X Axis"
slice1Display.DataAxesGrid.YTitle = "Y Axis"
slice1Display.DataAxesGrid.ZTitle = "Z Axis"
slice1Display.DataAxesGrid.XTitleFontFamily = "Arial"
slice1Display.DataAxesGrid.XTitleFontFile = ""
slice1Display.DataAxesGrid.XTitleBold = 0
slice1Display.DataAxesGrid.XTitleItalic = 0
slice1Display.DataAxesGrid.XTitleFontSize = 12
slice1Display.DataAxesGrid.XTitleShadow = 0
slice1Display.DataAxesGrid.XTitleOpacity = 1.0
slice1Display.DataAxesGrid.YTitleFontFamily = "Arial"
slice1Display.DataAxesGrid.YTitleFontFile = ""
slice1Display.DataAxesGrid.YTitleBold = 0
slice1Display.DataAxesGrid.YTitleItalic = 0
slice1Display.DataAxesGrid.YTitleFontSize = 12
slice1Display.DataAxesGrid.YTitleShadow = 0
slice1Display.DataAxesGrid.YTitleOpacity = 1.0
slice1Display.DataAxesGrid.ZTitleFontFamily = "Arial"
slice1Display.DataAxesGrid.ZTitleFontFile = ""
slice1Display.DataAxesGrid.ZTitleBold = 0
slice1Display.DataAxesGrid.ZTitleItalic = 0
slice1Display.DataAxesGrid.ZTitleFontSize = 12
slice1Display.DataAxesGrid.ZTitleShadow = 0
slice1Display.DataAxesGrid.ZTitleOpacity = 1.0
slice1Display.DataAxesGrid.FacesToRender = 63
slice1Display.DataAxesGrid.CullBackface = 0
slice1Display.DataAxesGrid.CullFrontface = 1
slice1Display.DataAxesGrid.ShowGrid = 0
slice1Display.DataAxesGrid.ShowEdges = 1
slice1Display.DataAxesGrid.ShowTicks = 1
slice1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
slice1Display.DataAxesGrid.AxesToLabel = 63
slice1Display.DataAxesGrid.XLabelFontFamily = "Arial"
slice1Display.DataAxesGrid.XLabelFontFile = ""
slice1Display.DataAxesGrid.XLabelBold = 0
slice1Display.DataAxesGrid.XLabelItalic = 0
slice1Display.DataAxesGrid.XLabelFontSize = 12
slice1Display.DataAxesGrid.XLabelShadow = 0
slice1Display.DataAxesGrid.XLabelOpacity = 1.0
slice1Display.DataAxesGrid.YLabelFontFamily = "Arial"
slice1Display.DataAxesGrid.YLabelFontFile = ""
slice1Display.DataAxesGrid.YLabelBold = 0
slice1Display.DataAxesGrid.YLabelItalic = 0
slice1Display.DataAxesGrid.YLabelFontSize = 12
slice1Display.DataAxesGrid.YLabelShadow = 0
slice1Display.DataAxesGrid.YLabelOpacity = 1.0
slice1Display.DataAxesGrid.ZLabelFontFamily = "Arial"
slice1Display.DataAxesGrid.ZLabelFontFile = ""
slice1Display.DataAxesGrid.ZLabelBold = 0
slice1Display.DataAxesGrid.ZLabelItalic = 0
slice1Display.DataAxesGrid.ZLabelFontSize = 12
slice1Display.DataAxesGrid.ZLabelShadow = 0
slice1Display.DataAxesGrid.ZLabelOpacity = 1.0
slice1Display.DataAxesGrid.XAxisNotation = "Mixed"
slice1Display.DataAxesGrid.XAxisPrecision = 2
slice1Display.DataAxesGrid.XAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.XAxisLabels = []
slice1Display.DataAxesGrid.YAxisNotation = "Mixed"
slice1Display.DataAxesGrid.YAxisPrecision = 2
slice1Display.DataAxesGrid.YAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.YAxisLabels = []
slice1Display.DataAxesGrid.ZAxisNotation = "Mixed"
slice1Display.DataAxesGrid.ZAxisPrecision = 2
slice1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
slice1Display.DataAxesGrid.ZAxisLabels = []
slice1Display.DataAxesGrid.UseCustomBounds = 0
slice1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
slice1Display.PolarAxes.Visibility = 0
slice1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
slice1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
slice1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
slice1Display.PolarAxes.EnableCustomRange = 0
slice1Display.PolarAxes.CustomRange = [0.0, 1.0]
slice1Display.PolarAxes.PolarAxisVisibility = 1
slice1Display.PolarAxes.RadialAxesVisibility = 1
slice1Display.PolarAxes.DrawRadialGridlines = 1
slice1Display.PolarAxes.PolarArcsVisibility = 1
slice1Display.PolarAxes.DrawPolarArcsGridlines = 1
slice1Display.PolarAxes.NumberOfRadialAxes = 0
slice1Display.PolarAxes.AutoSubdividePolarAxis = 1
slice1Display.PolarAxes.NumberOfPolarAxis = 0
slice1Display.PolarAxes.MinimumRadius = 0.0
slice1Display.PolarAxes.MinimumAngle = 0.0
slice1Display.PolarAxes.MaximumAngle = 90.0
slice1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
slice1Display.PolarAxes.Ratio = 1.0
slice1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
slice1Display.PolarAxes.PolarAxisTitleVisibility = 1
slice1Display.PolarAxes.PolarAxisTitle = "Radial Distance"
slice1Display.PolarAxes.PolarAxisTitleLocation = "Bottom"
slice1Display.PolarAxes.PolarLabelVisibility = 1
slice1Display.PolarAxes.PolarLabelFormat = "%-#6.3g"
slice1Display.PolarAxes.PolarLabelExponentLocation = "Labels"
slice1Display.PolarAxes.RadialLabelVisibility = 1
slice1Display.PolarAxes.RadialLabelFormat = "%-#3.1f"
slice1Display.PolarAxes.RadialLabelLocation = "Bottom"
slice1Display.PolarAxes.RadialUnitsVisibility = 1
slice1Display.PolarAxes.ScreenSize = 10.0
slice1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
slice1Display.PolarAxes.PolarAxisTitleFontFamily = "Arial"
slice1Display.PolarAxes.PolarAxisTitleFontFile = ""
slice1Display.PolarAxes.PolarAxisTitleBold = 0
slice1Display.PolarAxes.PolarAxisTitleItalic = 0
slice1Display.PolarAxes.PolarAxisTitleShadow = 0
slice1Display.PolarAxes.PolarAxisTitleFontSize = 12
slice1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
slice1Display.PolarAxes.PolarAxisLabelFontFamily = "Arial"
slice1Display.PolarAxes.PolarAxisLabelFontFile = ""
slice1Display.PolarAxes.PolarAxisLabelBold = 0
slice1Display.PolarAxes.PolarAxisLabelItalic = 0
slice1Display.PolarAxes.PolarAxisLabelShadow = 0
slice1Display.PolarAxes.PolarAxisLabelFontSize = 12
slice1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
slice1Display.PolarAxes.LastRadialAxisTextFontFamily = "Arial"
slice1Display.PolarAxes.LastRadialAxisTextFontFile = ""
slice1Display.PolarAxes.LastRadialAxisTextBold = 0
slice1Display.PolarAxes.LastRadialAxisTextItalic = 0
slice1Display.PolarAxes.LastRadialAxisTextShadow = 0
slice1Display.PolarAxes.LastRadialAxisTextFontSize = 12
slice1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = "Arial"
slice1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ""
slice1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
slice1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
slice1Display.PolarAxes.EnableDistanceLOD = 1
slice1Display.PolarAxes.DistanceLODThreshold = 0.7
slice1Display.PolarAxes.EnableViewAngleLOD = 1
slice1Display.PolarAxes.ViewAngleLODThreshold = 0.7
slice1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
slice1Display.PolarAxes.PolarTicksVisibility = 1
slice1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
slice1Display.PolarAxes.TickLocation = "Both"
slice1Display.PolarAxes.AxisTickVisibility = 1
slice1Display.PolarAxes.AxisMinorTickVisibility = 0
slice1Display.PolarAxes.ArcTickVisibility = 1
slice1Display.PolarAxes.ArcMinorTickVisibility = 0
slice1Display.PolarAxes.DeltaAngleMajor = 10.0
slice1Display.PolarAxes.DeltaAngleMinor = 5.0
slice1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
slice1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
slice1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
slice1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
slice1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
slice1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
slice1Display.PolarAxes.ArcMajorTickSize = 0.0
slice1Display.PolarAxes.ArcTickRatioSize = 0.3
slice1Display.PolarAxes.ArcMajorTickThickness = 1.0
slice1Display.PolarAxes.ArcTickRatioThickness = 0.5
slice1Display.PolarAxes.Use2DMode = 0
slice1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(aafoam, renderView1)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(slice1Display, ("POINTS", "s"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
slice1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 's'
sLUT = GetColorTransferFunction("s")
sLUT.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
sLUT.InterpretValuesAsCategories = 0
sLUT.AnnotationsInitialized = 0
sLUT.ShowCategoricalColorsinDataRangeOnly = 0
sLUT.RescaleOnVisibilityChange = 0
sLUT.EnableOpacityMapping = 0
sLUT.RGBPoints = [
    -8.445689445579774e-07,
    0.231373,
    0.298039,
    0.752941,
    0.2274766811872837,
    0.865003,
    0.865003,
    0.865003,
    0.45495420694351196,
    0.705882,
    0.0156863,
    0.14902,
]
sLUT.UseLogScale = 0
sLUT.ShowDataHistogram = 0
sLUT.AutomaticDataHistogramComputation = 0
sLUT.DataHistogramNumberOfBins = 10
sLUT.ColorSpace = "Diverging"
sLUT.UseBelowRangeColor = 0
sLUT.BelowRangeColor = [0.0, 0.0, 0.0]
sLUT.UseAboveRangeColor = 0
sLUT.AboveRangeColor = [0.5, 0.5, 0.5]
sLUT.NanColor = [1.0, 1.0, 0.0]
sLUT.NanOpacity = 1.0
sLUT.Discretize = 1
sLUT.NumberOfTableValues = 12
sLUT.ScalarRangeInitialized = 1.0
sLUT.HSVWrap = 0
sLUT.VectorComponent = 0
sLUT.VectorMode = "Magnitude"
sLUT.AllowDuplicateScalars = 1
sLUT.Annotations = []
sLUT.ActiveAnnotatedValues = []
sLUT.IndexedColors = []
sLUT.IndexedOpacities = []

# get opacity transfer function/opacity map for 's'
sPWF = GetOpacityTransferFunction("s")
sPWF.Points = [
    -8.445689445579774e-07,
    0.0,
    0.5,
    0.0,
    0.45495420694351196,
    1.0,
    0.5,
    0.0,
]
sPWF.AllowDuplicateScalars = 1
sPWF.UseLogScale = 0
sPWF.ScalarRangeInitialized = 1

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1.SliceType)

# reset view to fit data
renderView1.ResetCamera()

# Properties modified on animationScene1
animationScene1.AnimationTime = float(sys.argv[1])

# Properties modified on timeKeeper1
timeKeeper1.Time = float(sys.argv[1])

# Rescale transfer function
sLUT.RescaleTransferFunction(0.0, 0.15)

# Rescale transfer function
sPWF.RescaleTransferFunction(0.0, 0.15)

# create a new 'Surface Vectors'
surfaceVectors1 = SurfaceVectors(Input=slice1)
surfaceVectors1.SelectInputVectors = ["POINTS", "U"]
surfaceVectors1.ConstraintMode = "Parallel"

# Properties modified on surfaceVectors1
surfaceVectors1.SelectInputVectors = ["POINTS", "vorticityField"]

# show data in view
surfaceVectors1Display = Show(surfaceVectors1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
surfaceVectors1Display.Representation = "Surface"
surfaceVectors1Display.ColorArrayName = ["POINTS", "p"]
surfaceVectors1Display.LookupTable = pLUT
surfaceVectors1Display.MapScalars = 1
surfaceVectors1Display.MultiComponentsMapping = 0
surfaceVectors1Display.InterpolateScalarsBeforeMapping = 1
surfaceVectors1Display.Opacity = 1.0
surfaceVectors1Display.PointSize = 2.0
surfaceVectors1Display.LineWidth = 1.0
surfaceVectors1Display.RenderLinesAsTubes = 0
surfaceVectors1Display.RenderPointsAsSpheres = 0
surfaceVectors1Display.Interpolation = "Gouraud"
surfaceVectors1Display.Specular = 0.0
surfaceVectors1Display.SpecularColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.SpecularPower = 100.0
surfaceVectors1Display.Luminosity = 0.0
surfaceVectors1Display.Ambient = 0.0
surfaceVectors1Display.Diffuse = 1.0
surfaceVectors1Display.Roughness = 0.3
surfaceVectors1Display.Metallic = 0.0
surfaceVectors1Display.Texture = None
surfaceVectors1Display.RepeatTextures = 1
surfaceVectors1Display.InterpolateTextures = 0
surfaceVectors1Display.SeamlessU = 0
surfaceVectors1Display.SeamlessV = 0
surfaceVectors1Display.UseMipmapTextures = 0
surfaceVectors1Display.BaseColorTexture = None
surfaceVectors1Display.NormalTexture = None
surfaceVectors1Display.NormalScale = 1.0
surfaceVectors1Display.MaterialTexture = None
surfaceVectors1Display.OcclusionStrength = 1.0
surfaceVectors1Display.EmissiveTexture = None
surfaceVectors1Display.EmissiveFactor = [1.0, 1.0, 1.0]
surfaceVectors1Display.FlipTextures = 0
surfaceVectors1Display.BackfaceRepresentation = "Follow Frontface"
surfaceVectors1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.BackfaceOpacity = 1.0
surfaceVectors1Display.Position = [0.0, 0.0, 0.0]
surfaceVectors1Display.Scale = [1.0, 1.0, 1.0]
surfaceVectors1Display.Orientation = [0.0, 0.0, 0.0]
surfaceVectors1Display.Origin = [0.0, 0.0, 0.0]
surfaceVectors1Display.Pickable = 1
surfaceVectors1Display.Triangulate = 0
surfaceVectors1Display.UseShaderReplacements = 0
surfaceVectors1Display.ShaderReplacements = ""
surfaceVectors1Display.NonlinearSubdivisionLevel = 1
surfaceVectors1Display.UseDataPartitions = 0
surfaceVectors1Display.OSPRayUseScaleArray = 0
surfaceVectors1Display.OSPRayScaleArray = "p"
surfaceVectors1Display.OSPRayScaleFunction = "PiecewiseFunction"
surfaceVectors1Display.OSPRayMaterial = "None"
surfaceVectors1Display.Orient = 0
surfaceVectors1Display.OrientationMode = "Direction"
surfaceVectors1Display.SelectOrientationVectors = "vorticityField"
surfaceVectors1Display.Scaling = 0
surfaceVectors1Display.ScaleMode = "No Data Scaling Off"
surfaceVectors1Display.ScaleFactor = 0.002999929804354906
surfaceVectors1Display.SelectScaleArray = "p"
surfaceVectors1Display.GlyphType = "Arrow"
surfaceVectors1Display.UseGlyphTable = 0
surfaceVectors1Display.GlyphTableIndexArray = "p"
surfaceVectors1Display.UseCompositeGlyphTable = 0
surfaceVectors1Display.UseGlyphCullingAndLOD = 0
surfaceVectors1Display.LODValues = []
surfaceVectors1Display.ColorByLODIndex = 0
surfaceVectors1Display.GaussianRadius = 0.0001499964902177453
surfaceVectors1Display.ShaderPreset = "Sphere"
surfaceVectors1Display.CustomTriangleScale = 3
surfaceVectors1Display.CustomShader = """ // This custom shader code define a gaussian blur
// Please take a look into vtkSMPointGaussianRepresentation.cxx
// for other custom shader examples
//VTK::Color::Impl
  float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
  float gaussian = exp(-0.5*dist2);
  opacity = opacity*gaussian;
"""
surfaceVectors1Display.Emissive = 0
surfaceVectors1Display.ScaleByArray = 0
surfaceVectors1Display.SetScaleArray = ["POINTS", "p"]
surfaceVectors1Display.ScaleArrayComponent = ""
surfaceVectors1Display.UseScaleFunction = 1
surfaceVectors1Display.ScaleTransferFunction = "PiecewiseFunction"
surfaceVectors1Display.OpacityByArray = 0
surfaceVectors1Display.OpacityArray = ["POINTS", "p"]
surfaceVectors1Display.OpacityArrayComponent = ""
surfaceVectors1Display.OpacityTransferFunction = "PiecewiseFunction"
surfaceVectors1Display.DataAxesGrid = "GridAxesRepresentation"
surfaceVectors1Display.SelectionCellLabelBold = 0
surfaceVectors1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
surfaceVectors1Display.SelectionCellLabelFontFamily = "Arial"
surfaceVectors1Display.SelectionCellLabelFontFile = ""
surfaceVectors1Display.SelectionCellLabelFontSize = 18
surfaceVectors1Display.SelectionCellLabelItalic = 0
surfaceVectors1Display.SelectionCellLabelJustification = "Left"
surfaceVectors1Display.SelectionCellLabelOpacity = 1.0
surfaceVectors1Display.SelectionCellLabelShadow = 0
surfaceVectors1Display.SelectionPointLabelBold = 0
surfaceVectors1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
surfaceVectors1Display.SelectionPointLabelFontFamily = "Arial"
surfaceVectors1Display.SelectionPointLabelFontFile = ""
surfaceVectors1Display.SelectionPointLabelFontSize = 18
surfaceVectors1Display.SelectionPointLabelItalic = 0
surfaceVectors1Display.SelectionPointLabelJustification = "Left"
surfaceVectors1Display.SelectionPointLabelOpacity = 1.0
surfaceVectors1Display.SelectionPointLabelShadow = 0
surfaceVectors1Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
surfaceVectors1Display.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]
surfaceVectors1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
surfaceVectors1Display.GlyphType.TipResolution = 6
surfaceVectors1Display.GlyphType.TipRadius = 0.1
surfaceVectors1Display.GlyphType.TipLength = 0.35
surfaceVectors1Display.GlyphType.ShaftResolution = 6
surfaceVectors1Display.GlyphType.ShaftRadius = 0.03
surfaceVectors1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
surfaceVectors1Display.ScaleTransferFunction.Points = [
    -0.30084919929504395,
    0.0,
    0.5,
    0.0,
    -0.0472160279750824,
    1.0,
    0.5,
    0.0,
]
surfaceVectors1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
surfaceVectors1Display.OpacityTransferFunction.Points = [
    -0.30084919929504395,
    0.0,
    0.5,
    0.0,
    -0.0472160279750824,
    1.0,
    0.5,
    0.0,
]
surfaceVectors1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
surfaceVectors1Display.DataAxesGrid.XTitle = "X Axis"
surfaceVectors1Display.DataAxesGrid.YTitle = "Y Axis"
surfaceVectors1Display.DataAxesGrid.ZTitle = "Z Axis"
surfaceVectors1Display.DataAxesGrid.XTitleFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.XTitleFontFile = ""
surfaceVectors1Display.DataAxesGrid.XTitleBold = 0
surfaceVectors1Display.DataAxesGrid.XTitleItalic = 0
surfaceVectors1Display.DataAxesGrid.XTitleFontSize = 12
surfaceVectors1Display.DataAxesGrid.XTitleShadow = 0
surfaceVectors1Display.DataAxesGrid.XTitleOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.YTitleFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.YTitleFontFile = ""
surfaceVectors1Display.DataAxesGrid.YTitleBold = 0
surfaceVectors1Display.DataAxesGrid.YTitleItalic = 0
surfaceVectors1Display.DataAxesGrid.YTitleFontSize = 12
surfaceVectors1Display.DataAxesGrid.YTitleShadow = 0
surfaceVectors1Display.DataAxesGrid.YTitleOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.ZTitleFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.ZTitleFontFile = ""
surfaceVectors1Display.DataAxesGrid.ZTitleBold = 0
surfaceVectors1Display.DataAxesGrid.ZTitleItalic = 0
surfaceVectors1Display.DataAxesGrid.ZTitleFontSize = 12
surfaceVectors1Display.DataAxesGrid.ZTitleShadow = 0
surfaceVectors1Display.DataAxesGrid.ZTitleOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.FacesToRender = 63
surfaceVectors1Display.DataAxesGrid.CullBackface = 0
surfaceVectors1Display.DataAxesGrid.CullFrontface = 1
surfaceVectors1Display.DataAxesGrid.ShowGrid = 0
surfaceVectors1Display.DataAxesGrid.ShowEdges = 1
surfaceVectors1Display.DataAxesGrid.ShowTicks = 1
surfaceVectors1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
surfaceVectors1Display.DataAxesGrid.AxesToLabel = 63
surfaceVectors1Display.DataAxesGrid.XLabelFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.XLabelFontFile = ""
surfaceVectors1Display.DataAxesGrid.XLabelBold = 0
surfaceVectors1Display.DataAxesGrid.XLabelItalic = 0
surfaceVectors1Display.DataAxesGrid.XLabelFontSize = 12
surfaceVectors1Display.DataAxesGrid.XLabelShadow = 0
surfaceVectors1Display.DataAxesGrid.XLabelOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.YLabelFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.YLabelFontFile = ""
surfaceVectors1Display.DataAxesGrid.YLabelBold = 0
surfaceVectors1Display.DataAxesGrid.YLabelItalic = 0
surfaceVectors1Display.DataAxesGrid.YLabelFontSize = 12
surfaceVectors1Display.DataAxesGrid.YLabelShadow = 0
surfaceVectors1Display.DataAxesGrid.YLabelOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.ZLabelFontFamily = "Arial"
surfaceVectors1Display.DataAxesGrid.ZLabelFontFile = ""
surfaceVectors1Display.DataAxesGrid.ZLabelBold = 0
surfaceVectors1Display.DataAxesGrid.ZLabelItalic = 0
surfaceVectors1Display.DataAxesGrid.ZLabelFontSize = 12
surfaceVectors1Display.DataAxesGrid.ZLabelShadow = 0
surfaceVectors1Display.DataAxesGrid.ZLabelOpacity = 1.0
surfaceVectors1Display.DataAxesGrid.XAxisNotation = "Mixed"
surfaceVectors1Display.DataAxesGrid.XAxisPrecision = 2
surfaceVectors1Display.DataAxesGrid.XAxisUseCustomLabels = 0
surfaceVectors1Display.DataAxesGrid.XAxisLabels = []
surfaceVectors1Display.DataAxesGrid.YAxisNotation = "Mixed"
surfaceVectors1Display.DataAxesGrid.YAxisPrecision = 2
surfaceVectors1Display.DataAxesGrid.YAxisUseCustomLabels = 0
surfaceVectors1Display.DataAxesGrid.YAxisLabels = []
surfaceVectors1Display.DataAxesGrid.ZAxisNotation = "Mixed"
surfaceVectors1Display.DataAxesGrid.ZAxisPrecision = 2
surfaceVectors1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
surfaceVectors1Display.DataAxesGrid.ZAxisLabels = []
surfaceVectors1Display.DataAxesGrid.UseCustomBounds = 0
surfaceVectors1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
surfaceVectors1Display.PolarAxes.Visibility = 0
surfaceVectors1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
surfaceVectors1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
surfaceVectors1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
surfaceVectors1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
surfaceVectors1Display.PolarAxes.EnableCustomRange = 0
surfaceVectors1Display.PolarAxes.CustomRange = [0.0, 1.0]
surfaceVectors1Display.PolarAxes.PolarAxisVisibility = 1
surfaceVectors1Display.PolarAxes.RadialAxesVisibility = 1
surfaceVectors1Display.PolarAxes.DrawRadialGridlines = 1
surfaceVectors1Display.PolarAxes.PolarArcsVisibility = 1
surfaceVectors1Display.PolarAxes.DrawPolarArcsGridlines = 1
surfaceVectors1Display.PolarAxes.NumberOfRadialAxes = 0
surfaceVectors1Display.PolarAxes.AutoSubdividePolarAxis = 1
surfaceVectors1Display.PolarAxes.NumberOfPolarAxis = 0
surfaceVectors1Display.PolarAxes.MinimumRadius = 0.0
surfaceVectors1Display.PolarAxes.MinimumAngle = 0.0
surfaceVectors1Display.PolarAxes.MaximumAngle = 90.0
surfaceVectors1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
surfaceVectors1Display.PolarAxes.Ratio = 1.0
surfaceVectors1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
surfaceVectors1Display.PolarAxes.PolarAxisTitleVisibility = 1
surfaceVectors1Display.PolarAxes.PolarAxisTitle = "Radial Distance"
surfaceVectors1Display.PolarAxes.PolarAxisTitleLocation = "Bottom"
surfaceVectors1Display.PolarAxes.PolarLabelVisibility = 1
surfaceVectors1Display.PolarAxes.PolarLabelFormat = "%-#6.3g"
surfaceVectors1Display.PolarAxes.PolarLabelExponentLocation = "Labels"
surfaceVectors1Display.PolarAxes.RadialLabelVisibility = 1
surfaceVectors1Display.PolarAxes.RadialLabelFormat = "%-#3.1f"
surfaceVectors1Display.PolarAxes.RadialLabelLocation = "Bottom"
surfaceVectors1Display.PolarAxes.RadialUnitsVisibility = 1
surfaceVectors1Display.PolarAxes.ScreenSize = 10.0
surfaceVectors1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
surfaceVectors1Display.PolarAxes.PolarAxisTitleFontFamily = "Arial"
surfaceVectors1Display.PolarAxes.PolarAxisTitleFontFile = ""
surfaceVectors1Display.PolarAxes.PolarAxisTitleBold = 0
surfaceVectors1Display.PolarAxes.PolarAxisTitleItalic = 0
surfaceVectors1Display.PolarAxes.PolarAxisTitleShadow = 0
surfaceVectors1Display.PolarAxes.PolarAxisTitleFontSize = 12
surfaceVectors1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
surfaceVectors1Display.PolarAxes.PolarAxisLabelFontFamily = "Arial"
surfaceVectors1Display.PolarAxes.PolarAxisLabelFontFile = ""
surfaceVectors1Display.PolarAxes.PolarAxisLabelBold = 0
surfaceVectors1Display.PolarAxes.PolarAxisLabelItalic = 0
surfaceVectors1Display.PolarAxes.PolarAxisLabelShadow = 0
surfaceVectors1Display.PolarAxes.PolarAxisLabelFontSize = 12
surfaceVectors1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
surfaceVectors1Display.PolarAxes.LastRadialAxisTextFontFamily = "Arial"
surfaceVectors1Display.PolarAxes.LastRadialAxisTextFontFile = ""
surfaceVectors1Display.PolarAxes.LastRadialAxisTextBold = 0
surfaceVectors1Display.PolarAxes.LastRadialAxisTextItalic = 0
surfaceVectors1Display.PolarAxes.LastRadialAxisTextShadow = 0
surfaceVectors1Display.PolarAxes.LastRadialAxisTextFontSize = 12
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = "Arial"
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ""
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
surfaceVectors1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
surfaceVectors1Display.PolarAxes.EnableDistanceLOD = 1
surfaceVectors1Display.PolarAxes.DistanceLODThreshold = 0.7
surfaceVectors1Display.PolarAxes.EnableViewAngleLOD = 1
surfaceVectors1Display.PolarAxes.ViewAngleLODThreshold = 0.7
surfaceVectors1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
surfaceVectors1Display.PolarAxes.PolarTicksVisibility = 1
surfaceVectors1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
surfaceVectors1Display.PolarAxes.TickLocation = "Both"
surfaceVectors1Display.PolarAxes.AxisTickVisibility = 1
surfaceVectors1Display.PolarAxes.AxisMinorTickVisibility = 0
surfaceVectors1Display.PolarAxes.ArcTickVisibility = 1
surfaceVectors1Display.PolarAxes.ArcMinorTickVisibility = 0
surfaceVectors1Display.PolarAxes.DeltaAngleMajor = 10.0
surfaceVectors1Display.PolarAxes.DeltaAngleMinor = 5.0
surfaceVectors1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
surfaceVectors1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
surfaceVectors1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
surfaceVectors1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
surfaceVectors1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
surfaceVectors1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
surfaceVectors1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
surfaceVectors1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
surfaceVectors1Display.PolarAxes.ArcMajorTickSize = 0.0
surfaceVectors1Display.PolarAxes.ArcTickRatioSize = 0.3
surfaceVectors1Display.PolarAxes.ArcMajorTickThickness = 1.0
surfaceVectors1Display.PolarAxes.ArcTickRatioThickness = 0.5
surfaceVectors1Display.PolarAxes.Use2DMode = 0
surfaceVectors1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(slice1, renderView1)

# show color bar/color legend
surfaceVectors1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(surfaceVectors1Display, ("POINTS", "vorticityField", "Magnitude"))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
surfaceVectors1Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
surfaceVectors1Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'vorticityField'
vorticityFieldLUT = GetColorTransferFunction("vorticityField")
vorticityFieldLUT.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
vorticityFieldLUT.InterpretValuesAsCategories = 0
vorticityFieldLUT.AnnotationsInitialized = 0
vorticityFieldLUT.ShowCategoricalColorsinDataRangeOnly = 0
vorticityFieldLUT.RescaleOnVisibilityChange = 0
vorticityFieldLUT.EnableOpacityMapping = 0
vorticityFieldLUT.RGBPoints = [
    0.3821501140585807,
    0.231373,
    0.298039,
    0.752941,
    218.89495103418685,
    0.865003,
    0.865003,
    0.865003,
    437.4077519543151,
    0.705882,
    0.0156863,
    0.14902,
]
vorticityFieldLUT.UseLogScale = 0
vorticityFieldLUT.ShowDataHistogram = 0
vorticityFieldLUT.AutomaticDataHistogramComputation = 0
vorticityFieldLUT.DataHistogramNumberOfBins = 10
vorticityFieldLUT.ColorSpace = "Diverging"
vorticityFieldLUT.UseBelowRangeColor = 0
vorticityFieldLUT.BelowRangeColor = [0.0, 0.0, 0.0]
vorticityFieldLUT.UseAboveRangeColor = 0
vorticityFieldLUT.AboveRangeColor = [0.5, 0.5, 0.5]
vorticityFieldLUT.NanColor = [1.0, 1.0, 0.0]
vorticityFieldLUT.NanOpacity = 1.0
vorticityFieldLUT.Discretize = 1
vorticityFieldLUT.NumberOfTableValues = 256
vorticityFieldLUT.ScalarRangeInitialized = 1.0
vorticityFieldLUT.HSVWrap = 0
vorticityFieldLUT.VectorComponent = 0
vorticityFieldLUT.VectorMode = "Magnitude"
vorticityFieldLUT.AllowDuplicateScalars = 1
vorticityFieldLUT.Annotations = []
vorticityFieldLUT.ActiveAnnotatedValues = []
vorticityFieldLUT.IndexedColors = []
vorticityFieldLUT.IndexedOpacities = []

# get opacity transfer function/opacity map for 'vorticityField'
vorticityFieldPWF = GetOpacityTransferFunction("vorticityField")
vorticityFieldPWF.Points = [
    0.3821501140585807,
    0.0,
    0.5,
    0.0,
    437.4077519543151,
    1.0,
    0.5,
    0.0,
]
vorticityFieldPWF.AllowDuplicateScalars = 1
vorticityFieldPWF.UseLogScale = 0
vorticityFieldPWF.ScalarRangeInitialized = 1

# create a new 'Mask Points'
maskPoints1 = MaskPoints(Input=surfaceVectors1)
maskPoints1.OnRatio = 2
maskPoints1.MaximumNumberofPoints = 5000
maskPoints1.ProportionallyDistributeMaximumNumberOfPoints = 0
maskPoints1.Offset = 0
maskPoints1.RandomSampling = 0
maskPoints1.RandomSamplingMode = "Randomized Id Strides"
maskPoints1.GenerateVertices = 0
maskPoints1.SingleVertexPerCell = 0

# Properties modified on maskPoints1
maskPoints1.OnRatio = 15
maskPoints1.MaximumNumberofPoints = 1000

# show data in view
maskPoints1Display = Show(maskPoints1, renderView1, "GeometryRepresentation")

# trace defaults for the display properties.
maskPoints1Display.Representation = "Surface"
maskPoints1Display.ColorArrayName = ["POINTS", "p"]
maskPoints1Display.LookupTable = pLUT
maskPoints1Display.MapScalars = 1
maskPoints1Display.MultiComponentsMapping = 0
maskPoints1Display.InterpolateScalarsBeforeMapping = 1
maskPoints1Display.Opacity = 1.0
maskPoints1Display.PointSize = 2.0
maskPoints1Display.LineWidth = 1.0
maskPoints1Display.RenderLinesAsTubes = 0
maskPoints1Display.RenderPointsAsSpheres = 0
maskPoints1Display.Interpolation = "Gouraud"
maskPoints1Display.Specular = 0.0
maskPoints1Display.SpecularColor = [1.0, 1.0, 1.0]
maskPoints1Display.SpecularPower = 100.0
maskPoints1Display.Luminosity = 0.0
maskPoints1Display.Ambient = 0.0
maskPoints1Display.Diffuse = 1.0
maskPoints1Display.Roughness = 0.3
maskPoints1Display.Metallic = 0.0
maskPoints1Display.Texture = None
maskPoints1Display.RepeatTextures = 1
maskPoints1Display.InterpolateTextures = 0
maskPoints1Display.SeamlessU = 0
maskPoints1Display.SeamlessV = 0
maskPoints1Display.UseMipmapTextures = 0
maskPoints1Display.BaseColorTexture = None
maskPoints1Display.NormalTexture = None
maskPoints1Display.NormalScale = 1.0
maskPoints1Display.MaterialTexture = None
maskPoints1Display.OcclusionStrength = 1.0
maskPoints1Display.EmissiveTexture = None
maskPoints1Display.EmissiveFactor = [1.0, 1.0, 1.0]
maskPoints1Display.FlipTextures = 0
maskPoints1Display.BackfaceRepresentation = "Follow Frontface"
maskPoints1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
maskPoints1Display.BackfaceOpacity = 1.0
maskPoints1Display.Position = [0.0, 0.0, 0.0]
maskPoints1Display.Scale = [1.0, 1.0, 1.0]
maskPoints1Display.Orientation = [0.0, 0.0, 0.0]
maskPoints1Display.Origin = [0.0, 0.0, 0.0]
maskPoints1Display.Pickable = 1
maskPoints1Display.Triangulate = 0
maskPoints1Display.UseShaderReplacements = 0
maskPoints1Display.ShaderReplacements = ""
maskPoints1Display.NonlinearSubdivisionLevel = 1
maskPoints1Display.UseDataPartitions = 0
maskPoints1Display.OSPRayUseScaleArray = 0
maskPoints1Display.OSPRayScaleArray = "p"
maskPoints1Display.OSPRayScaleFunction = "PiecewiseFunction"
maskPoints1Display.OSPRayMaterial = "None"
maskPoints1Display.Orient = 0
maskPoints1Display.OrientationMode = "Direction"
maskPoints1Display.SelectOrientationVectors = "vorticityField"
maskPoints1Display.Scaling = 0
maskPoints1Display.ScaleMode = "No Data Scaling Off"
maskPoints1Display.ScaleFactor = 0.0029181145131587983
maskPoints1Display.SelectScaleArray = "p"
maskPoints1Display.GlyphType = "Arrow"
maskPoints1Display.UseGlyphTable = 0
maskPoints1Display.GlyphTableIndexArray = "p"
maskPoints1Display.UseCompositeGlyphTable = 0
maskPoints1Display.UseGlyphCullingAndLOD = 0
maskPoints1Display.LODValues = []
maskPoints1Display.ColorByLODIndex = 0
maskPoints1Display.GaussianRadius = 0.00014590572565793993
maskPoints1Display.ShaderPreset = "Sphere"
maskPoints1Display.CustomTriangleScale = 3
maskPoints1Display.CustomShader = """ // This custom shader code define a gaussian blur
// Please take a look into vtkSMPointGaussianRepresentation.cxx
// for other custom shader examples
//VTK::Color::Impl
  float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
  float gaussian = exp(-0.5*dist2);
  opacity = opacity*gaussian;
"""
maskPoints1Display.Emissive = 0
maskPoints1Display.ScaleByArray = 0
maskPoints1Display.SetScaleArray = ["POINTS", "p"]
maskPoints1Display.ScaleArrayComponent = ""
maskPoints1Display.UseScaleFunction = 1
maskPoints1Display.ScaleTransferFunction = "PiecewiseFunction"
maskPoints1Display.OpacityByArray = 0
maskPoints1Display.OpacityArray = ["POINTS", "p"]
maskPoints1Display.OpacityArrayComponent = ""
maskPoints1Display.OpacityTransferFunction = "PiecewiseFunction"
maskPoints1Display.DataAxesGrid = "GridAxesRepresentation"
maskPoints1Display.SelectionCellLabelBold = 0
maskPoints1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
maskPoints1Display.SelectionCellLabelFontFamily = "Arial"
maskPoints1Display.SelectionCellLabelFontFile = ""
maskPoints1Display.SelectionCellLabelFontSize = 18
maskPoints1Display.SelectionCellLabelItalic = 0
maskPoints1Display.SelectionCellLabelJustification = "Left"
maskPoints1Display.SelectionCellLabelOpacity = 1.0
maskPoints1Display.SelectionCellLabelShadow = 0
maskPoints1Display.SelectionPointLabelBold = 0
maskPoints1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
maskPoints1Display.SelectionPointLabelFontFamily = "Arial"
maskPoints1Display.SelectionPointLabelFontFile = ""
maskPoints1Display.SelectionPointLabelFontSize = 18
maskPoints1Display.SelectionPointLabelItalic = 0
maskPoints1Display.SelectionPointLabelJustification = "Left"
maskPoints1Display.SelectionPointLabelOpacity = 1.0
maskPoints1Display.SelectionPointLabelShadow = 0
maskPoints1Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
maskPoints1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
maskPoints1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
maskPoints1Display.GlyphType.TipResolution = 6
maskPoints1Display.GlyphType.TipRadius = 0.1
maskPoints1Display.GlyphType.TipLength = 0.35
maskPoints1Display.GlyphType.ShaftResolution = 6
maskPoints1Display.GlyphType.ShaftRadius = 0.03
maskPoints1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
maskPoints1Display.ScaleTransferFunction.Points = [
    -0.3008255064487457,
    0.0,
    0.5,
    0.0,
    -0.04750198870897293,
    1.0,
    0.5,
    0.0,
]
maskPoints1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
maskPoints1Display.OpacityTransferFunction.Points = [
    -0.3008255064487457,
    0.0,
    0.5,
    0.0,
    -0.04750198870897293,
    1.0,
    0.5,
    0.0,
]
maskPoints1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
maskPoints1Display.DataAxesGrid.XTitle = "X Axis"
maskPoints1Display.DataAxesGrid.YTitle = "Y Axis"
maskPoints1Display.DataAxesGrid.ZTitle = "Z Axis"
maskPoints1Display.DataAxesGrid.XTitleFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.XTitleFontFile = ""
maskPoints1Display.DataAxesGrid.XTitleBold = 0
maskPoints1Display.DataAxesGrid.XTitleItalic = 0
maskPoints1Display.DataAxesGrid.XTitleFontSize = 12
maskPoints1Display.DataAxesGrid.XTitleShadow = 0
maskPoints1Display.DataAxesGrid.XTitleOpacity = 1.0
maskPoints1Display.DataAxesGrid.YTitleFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.YTitleFontFile = ""
maskPoints1Display.DataAxesGrid.YTitleBold = 0
maskPoints1Display.DataAxesGrid.YTitleItalic = 0
maskPoints1Display.DataAxesGrid.YTitleFontSize = 12
maskPoints1Display.DataAxesGrid.YTitleShadow = 0
maskPoints1Display.DataAxesGrid.YTitleOpacity = 1.0
maskPoints1Display.DataAxesGrid.ZTitleFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.ZTitleFontFile = ""
maskPoints1Display.DataAxesGrid.ZTitleBold = 0
maskPoints1Display.DataAxesGrid.ZTitleItalic = 0
maskPoints1Display.DataAxesGrid.ZTitleFontSize = 12
maskPoints1Display.DataAxesGrid.ZTitleShadow = 0
maskPoints1Display.DataAxesGrid.ZTitleOpacity = 1.0
maskPoints1Display.DataAxesGrid.FacesToRender = 63
maskPoints1Display.DataAxesGrid.CullBackface = 0
maskPoints1Display.DataAxesGrid.CullFrontface = 1
maskPoints1Display.DataAxesGrid.ShowGrid = 0
maskPoints1Display.DataAxesGrid.ShowEdges = 1
maskPoints1Display.DataAxesGrid.ShowTicks = 1
maskPoints1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
maskPoints1Display.DataAxesGrid.AxesToLabel = 63
maskPoints1Display.DataAxesGrid.XLabelFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.XLabelFontFile = ""
maskPoints1Display.DataAxesGrid.XLabelBold = 0
maskPoints1Display.DataAxesGrid.XLabelItalic = 0
maskPoints1Display.DataAxesGrid.XLabelFontSize = 12
maskPoints1Display.DataAxesGrid.XLabelShadow = 0
maskPoints1Display.DataAxesGrid.XLabelOpacity = 1.0
maskPoints1Display.DataAxesGrid.YLabelFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.YLabelFontFile = ""
maskPoints1Display.DataAxesGrid.YLabelBold = 0
maskPoints1Display.DataAxesGrid.YLabelItalic = 0
maskPoints1Display.DataAxesGrid.YLabelFontSize = 12
maskPoints1Display.DataAxesGrid.YLabelShadow = 0
maskPoints1Display.DataAxesGrid.YLabelOpacity = 1.0
maskPoints1Display.DataAxesGrid.ZLabelFontFamily = "Arial"
maskPoints1Display.DataAxesGrid.ZLabelFontFile = ""
maskPoints1Display.DataAxesGrid.ZLabelBold = 0
maskPoints1Display.DataAxesGrid.ZLabelItalic = 0
maskPoints1Display.DataAxesGrid.ZLabelFontSize = 12
maskPoints1Display.DataAxesGrid.ZLabelShadow = 0
maskPoints1Display.DataAxesGrid.ZLabelOpacity = 1.0
maskPoints1Display.DataAxesGrid.XAxisNotation = "Mixed"
maskPoints1Display.DataAxesGrid.XAxisPrecision = 2
maskPoints1Display.DataAxesGrid.XAxisUseCustomLabels = 0
maskPoints1Display.DataAxesGrid.XAxisLabels = []
maskPoints1Display.DataAxesGrid.YAxisNotation = "Mixed"
maskPoints1Display.DataAxesGrid.YAxisPrecision = 2
maskPoints1Display.DataAxesGrid.YAxisUseCustomLabels = 0
maskPoints1Display.DataAxesGrid.YAxisLabels = []
maskPoints1Display.DataAxesGrid.ZAxisNotation = "Mixed"
maskPoints1Display.DataAxesGrid.ZAxisPrecision = 2
maskPoints1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
maskPoints1Display.DataAxesGrid.ZAxisLabels = []
maskPoints1Display.DataAxesGrid.UseCustomBounds = 0
maskPoints1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
maskPoints1Display.PolarAxes.Visibility = 0
maskPoints1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
maskPoints1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
maskPoints1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
maskPoints1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
maskPoints1Display.PolarAxes.EnableCustomRange = 0
maskPoints1Display.PolarAxes.CustomRange = [0.0, 1.0]
maskPoints1Display.PolarAxes.PolarAxisVisibility = 1
maskPoints1Display.PolarAxes.RadialAxesVisibility = 1
maskPoints1Display.PolarAxes.DrawRadialGridlines = 1
maskPoints1Display.PolarAxes.PolarArcsVisibility = 1
maskPoints1Display.PolarAxes.DrawPolarArcsGridlines = 1
maskPoints1Display.PolarAxes.NumberOfRadialAxes = 0
maskPoints1Display.PolarAxes.AutoSubdividePolarAxis = 1
maskPoints1Display.PolarAxes.NumberOfPolarAxis = 0
maskPoints1Display.PolarAxes.MinimumRadius = 0.0
maskPoints1Display.PolarAxes.MinimumAngle = 0.0
maskPoints1Display.PolarAxes.MaximumAngle = 90.0
maskPoints1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
maskPoints1Display.PolarAxes.Ratio = 1.0
maskPoints1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
maskPoints1Display.PolarAxes.PolarAxisTitleVisibility = 1
maskPoints1Display.PolarAxes.PolarAxisTitle = "Radial Distance"
maskPoints1Display.PolarAxes.PolarAxisTitleLocation = "Bottom"
maskPoints1Display.PolarAxes.PolarLabelVisibility = 1
maskPoints1Display.PolarAxes.PolarLabelFormat = "%-#6.3g"
maskPoints1Display.PolarAxes.PolarLabelExponentLocation = "Labels"
maskPoints1Display.PolarAxes.RadialLabelVisibility = 1
maskPoints1Display.PolarAxes.RadialLabelFormat = "%-#3.1f"
maskPoints1Display.PolarAxes.RadialLabelLocation = "Bottom"
maskPoints1Display.PolarAxes.RadialUnitsVisibility = 1
maskPoints1Display.PolarAxes.ScreenSize = 10.0
maskPoints1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
maskPoints1Display.PolarAxes.PolarAxisTitleFontFamily = "Arial"
maskPoints1Display.PolarAxes.PolarAxisTitleFontFile = ""
maskPoints1Display.PolarAxes.PolarAxisTitleBold = 0
maskPoints1Display.PolarAxes.PolarAxisTitleItalic = 0
maskPoints1Display.PolarAxes.PolarAxisTitleShadow = 0
maskPoints1Display.PolarAxes.PolarAxisTitleFontSize = 12
maskPoints1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
maskPoints1Display.PolarAxes.PolarAxisLabelFontFamily = "Arial"
maskPoints1Display.PolarAxes.PolarAxisLabelFontFile = ""
maskPoints1Display.PolarAxes.PolarAxisLabelBold = 0
maskPoints1Display.PolarAxes.PolarAxisLabelItalic = 0
maskPoints1Display.PolarAxes.PolarAxisLabelShadow = 0
maskPoints1Display.PolarAxes.PolarAxisLabelFontSize = 12
maskPoints1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
maskPoints1Display.PolarAxes.LastRadialAxisTextFontFamily = "Arial"
maskPoints1Display.PolarAxes.LastRadialAxisTextFontFile = ""
maskPoints1Display.PolarAxes.LastRadialAxisTextBold = 0
maskPoints1Display.PolarAxes.LastRadialAxisTextItalic = 0
maskPoints1Display.PolarAxes.LastRadialAxisTextShadow = 0
maskPoints1Display.PolarAxes.LastRadialAxisTextFontSize = 12
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = "Arial"
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ""
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
maskPoints1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
maskPoints1Display.PolarAxes.EnableDistanceLOD = 1
maskPoints1Display.PolarAxes.DistanceLODThreshold = 0.7
maskPoints1Display.PolarAxes.EnableViewAngleLOD = 1
maskPoints1Display.PolarAxes.ViewAngleLODThreshold = 0.7
maskPoints1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
maskPoints1Display.PolarAxes.PolarTicksVisibility = 1
maskPoints1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
maskPoints1Display.PolarAxes.TickLocation = "Both"
maskPoints1Display.PolarAxes.AxisTickVisibility = 1
maskPoints1Display.PolarAxes.AxisMinorTickVisibility = 0
maskPoints1Display.PolarAxes.ArcTickVisibility = 1
maskPoints1Display.PolarAxes.ArcMinorTickVisibility = 0
maskPoints1Display.PolarAxes.DeltaAngleMajor = 10.0
maskPoints1Display.PolarAxes.DeltaAngleMinor = 5.0
maskPoints1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
maskPoints1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
maskPoints1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
maskPoints1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
maskPoints1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
maskPoints1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
maskPoints1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
maskPoints1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
maskPoints1Display.PolarAxes.ArcMajorTickSize = 0.0
maskPoints1Display.PolarAxes.ArcTickRatioSize = 0.3
maskPoints1Display.PolarAxes.ArcMajorTickThickness = 1.0
maskPoints1Display.PolarAxes.ArcTickRatioThickness = 0.5
maskPoints1Display.PolarAxes.Use2DMode = 0
maskPoints1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(surfaceVectors1, renderView1)

# show color bar/color legend
maskPoints1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# turn off scalar coloring
ColorBy(maskPoints1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# create a new 'Stream Tracer With Custom Source'
streamTracerWithCustomSource1 = StreamTracerWithCustomSource(
    Input=surfaceVectors1, SeedSource=maskPoints1
)
streamTracerWithCustomSource1.Vectors = ["POINTS", "vorticityField"]
streamTracerWithCustomSource1.InterpolatorType = "Interpolator with Point Locator"
streamTracerWithCustomSource1.SurfaceStreamlines = 0
streamTracerWithCustomSource1.IntegrationDirection = "BOTH"
streamTracerWithCustomSource1.IntegratorType = "Runge-Kutta 4-5"
streamTracerWithCustomSource1.IntegrationStepUnit = "Cell Length"
streamTracerWithCustomSource1.InitialStepLength = 0.2
streamTracerWithCustomSource1.MinimumStepLength = 0.01
streamTracerWithCustomSource1.MaximumStepLength = 0.5
streamTracerWithCustomSource1.MaximumSteps = 2000
streamTracerWithCustomSource1.MaximumStreamlineLength = 0.02999929804354906
streamTracerWithCustomSource1.TerminalSpeed = 1e-12
streamTracerWithCustomSource1.MaximumError = 1e-06
streamTracerWithCustomSource1.ComputeVorticity = 1

# show data in view
streamTracerWithCustomSource1Display = Show(
    streamTracerWithCustomSource1, renderView1, "GeometryRepresentation"
)

# trace defaults for the display properties.
streamTracerWithCustomSource1Display.Representation = "Surface"
streamTracerWithCustomSource1Display.ColorArrayName = ["POINTS", "p"]
streamTracerWithCustomSource1Display.LookupTable = pLUT
streamTracerWithCustomSource1Display.MapScalars = 1
streamTracerWithCustomSource1Display.MultiComponentsMapping = 0
streamTracerWithCustomSource1Display.InterpolateScalarsBeforeMapping = 1
streamTracerWithCustomSource1Display.Opacity = 1.0
streamTracerWithCustomSource1Display.PointSize = 2.0
streamTracerWithCustomSource1Display.LineWidth = 1.0
streamTracerWithCustomSource1Display.RenderLinesAsTubes = 0
streamTracerWithCustomSource1Display.RenderPointsAsSpheres = 0
streamTracerWithCustomSource1Display.Interpolation = "Gouraud"
streamTracerWithCustomSource1Display.Specular = 0.0
streamTracerWithCustomSource1Display.SpecularColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.SpecularPower = 100.0
streamTracerWithCustomSource1Display.Luminosity = 0.0
streamTracerWithCustomSource1Display.Ambient = 0.0
streamTracerWithCustomSource1Display.Diffuse = 1.0
streamTracerWithCustomSource1Display.Roughness = 0.3
streamTracerWithCustomSource1Display.Metallic = 0.0
streamTracerWithCustomSource1Display.Texture = None
streamTracerWithCustomSource1Display.RepeatTextures = 1
streamTracerWithCustomSource1Display.InterpolateTextures = 0
streamTracerWithCustomSource1Display.SeamlessU = 0
streamTracerWithCustomSource1Display.SeamlessV = 0
streamTracerWithCustomSource1Display.UseMipmapTextures = 0
streamTracerWithCustomSource1Display.BaseColorTexture = None
streamTracerWithCustomSource1Display.NormalTexture = None
streamTracerWithCustomSource1Display.NormalScale = 1.0
streamTracerWithCustomSource1Display.MaterialTexture = None
streamTracerWithCustomSource1Display.OcclusionStrength = 1.0
streamTracerWithCustomSource1Display.EmissiveTexture = None
streamTracerWithCustomSource1Display.EmissiveFactor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.FlipTextures = 0
streamTracerWithCustomSource1Display.BackfaceRepresentation = "Follow Frontface"
streamTracerWithCustomSource1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.BackfaceOpacity = 1.0
streamTracerWithCustomSource1Display.Position = [0.0, 0.0, 0.0]
streamTracerWithCustomSource1Display.Scale = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.Orientation = [0.0, 0.0, 0.0]
streamTracerWithCustomSource1Display.Origin = [0.0, 0.0, 0.0]
streamTracerWithCustomSource1Display.Pickable = 1
streamTracerWithCustomSource1Display.Triangulate = 0
streamTracerWithCustomSource1Display.UseShaderReplacements = 0
streamTracerWithCustomSource1Display.ShaderReplacements = ""
streamTracerWithCustomSource1Display.NonlinearSubdivisionLevel = 1
streamTracerWithCustomSource1Display.UseDataPartitions = 0
streamTracerWithCustomSource1Display.OSPRayUseScaleArray = 0
streamTracerWithCustomSource1Display.OSPRayScaleArray = "p"
streamTracerWithCustomSource1Display.OSPRayScaleFunction = "PiecewiseFunction"
streamTracerWithCustomSource1Display.OSPRayMaterial = "None"
streamTracerWithCustomSource1Display.Orient = 0
streamTracerWithCustomSource1Display.OrientationMode = "Direction"
streamTracerWithCustomSource1Display.SelectOrientationVectors = "Normals"
streamTracerWithCustomSource1Display.Scaling = 0
streamTracerWithCustomSource1Display.ScaleMode = "No Data Scaling Off"
streamTracerWithCustomSource1Display.ScaleFactor = 0.002978736534714699
streamTracerWithCustomSource1Display.SelectScaleArray = "p"
streamTracerWithCustomSource1Display.GlyphType = "Arrow"
streamTracerWithCustomSource1Display.UseGlyphTable = 0
streamTracerWithCustomSource1Display.GlyphTableIndexArray = "p"
streamTracerWithCustomSource1Display.UseCompositeGlyphTable = 0
streamTracerWithCustomSource1Display.UseGlyphCullingAndLOD = 0
streamTracerWithCustomSource1Display.LODValues = []
streamTracerWithCustomSource1Display.ColorByLODIndex = 0
streamTracerWithCustomSource1Display.GaussianRadius = 0.00014893682673573495
streamTracerWithCustomSource1Display.ShaderPreset = "Sphere"
streamTracerWithCustomSource1Display.CustomTriangleScale = 3
streamTracerWithCustomSource1Display.CustomShader = """ // This custom shader code define a gaussian blur
// Please take a look into vtkSMPointGaussianRepresentation.cxx
// for other custom shader examples
//VTK::Color::Impl
  float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
  float gaussian = exp(-0.5*dist2);
  opacity = opacity*gaussian;
"""
streamTracerWithCustomSource1Display.Emissive = 0
streamTracerWithCustomSource1Display.ScaleByArray = 0
streamTracerWithCustomSource1Display.SetScaleArray = ["POINTS", "p"]
streamTracerWithCustomSource1Display.ScaleArrayComponent = ""
streamTracerWithCustomSource1Display.UseScaleFunction = 1
streamTracerWithCustomSource1Display.ScaleTransferFunction = "PiecewiseFunction"
streamTracerWithCustomSource1Display.OpacityByArray = 0
streamTracerWithCustomSource1Display.OpacityArray = ["POINTS", "p"]
streamTracerWithCustomSource1Display.OpacityArrayComponent = ""
streamTracerWithCustomSource1Display.OpacityTransferFunction = "PiecewiseFunction"
streamTracerWithCustomSource1Display.DataAxesGrid = "GridAxesRepresentation"
streamTracerWithCustomSource1Display.SelectionCellLabelBold = 0
streamTracerWithCustomSource1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
streamTracerWithCustomSource1Display.SelectionCellLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.SelectionCellLabelFontFile = ""
streamTracerWithCustomSource1Display.SelectionCellLabelFontSize = 18
streamTracerWithCustomSource1Display.SelectionCellLabelItalic = 0
streamTracerWithCustomSource1Display.SelectionCellLabelJustification = "Left"
streamTracerWithCustomSource1Display.SelectionCellLabelOpacity = 1.0
streamTracerWithCustomSource1Display.SelectionCellLabelShadow = 0
streamTracerWithCustomSource1Display.SelectionPointLabelBold = 0
streamTracerWithCustomSource1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
streamTracerWithCustomSource1Display.SelectionPointLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.SelectionPointLabelFontFile = ""
streamTracerWithCustomSource1Display.SelectionPointLabelFontSize = 18
streamTracerWithCustomSource1Display.SelectionPointLabelItalic = 0
streamTracerWithCustomSource1Display.SelectionPointLabelJustification = "Left"
streamTracerWithCustomSource1Display.SelectionPointLabelOpacity = 1.0
streamTracerWithCustomSource1Display.SelectionPointLabelShadow = 0
streamTracerWithCustomSource1Display.PolarAxes = "PolarAxesRepresentation"

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
streamTracerWithCustomSource1Display.OSPRayScaleFunction.Points = [
    0.0,
    0.0,
    0.5,
    0.0,
    1.0,
    1.0,
    0.5,
    0.0,
]
streamTracerWithCustomSource1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
streamTracerWithCustomSource1Display.GlyphType.TipResolution = 6
streamTracerWithCustomSource1Display.GlyphType.TipRadius = 0.1
streamTracerWithCustomSource1Display.GlyphType.TipLength = 0.35
streamTracerWithCustomSource1Display.GlyphType.ShaftResolution = 6
streamTracerWithCustomSource1Display.GlyphType.ShaftRadius = 0.03
streamTracerWithCustomSource1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
streamTracerWithCustomSource1Display.ScaleTransferFunction.Points = [
    -0.3008274734020233,
    0.0,
    0.5,
    0.0,
    -0.047243405133485794,
    1.0,
    0.5,
    0.0,
]
streamTracerWithCustomSource1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
streamTracerWithCustomSource1Display.OpacityTransferFunction.Points = [
    -0.3008274734020233,
    0.0,
    0.5,
    0.0,
    -0.047243405133485794,
    1.0,
    0.5,
    0.0,
]
streamTracerWithCustomSource1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
streamTracerWithCustomSource1Display.DataAxesGrid.XTitle = "X Axis"
streamTracerWithCustomSource1Display.DataAxesGrid.YTitle = "Y Axis"
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitle = "Z Axis"
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XTitleOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YTitleOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZTitleOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.FacesToRender = 63
streamTracerWithCustomSource1Display.DataAxesGrid.CullBackface = 0
streamTracerWithCustomSource1Display.DataAxesGrid.CullFrontface = 1
streamTracerWithCustomSource1Display.DataAxesGrid.ShowGrid = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ShowEdges = 1
streamTracerWithCustomSource1Display.DataAxesGrid.ShowTicks = 1
streamTracerWithCustomSource1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
streamTracerWithCustomSource1Display.DataAxesGrid.AxesToLabel = 63
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XLabelOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YLabelOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelFontFile = ""
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelBold = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelItalic = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelFontSize = 12
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelShadow = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZLabelOpacity = 1.0
streamTracerWithCustomSource1Display.DataAxesGrid.XAxisNotation = "Mixed"
streamTracerWithCustomSource1Display.DataAxesGrid.XAxisPrecision = 2
streamTracerWithCustomSource1Display.DataAxesGrid.XAxisUseCustomLabels = 0
streamTracerWithCustomSource1Display.DataAxesGrid.XAxisLabels = []
streamTracerWithCustomSource1Display.DataAxesGrid.YAxisNotation = "Mixed"
streamTracerWithCustomSource1Display.DataAxesGrid.YAxisPrecision = 2
streamTracerWithCustomSource1Display.DataAxesGrid.YAxisUseCustomLabels = 0
streamTracerWithCustomSource1Display.DataAxesGrid.YAxisLabels = []
streamTracerWithCustomSource1Display.DataAxesGrid.ZAxisNotation = "Mixed"
streamTracerWithCustomSource1Display.DataAxesGrid.ZAxisPrecision = 2
streamTracerWithCustomSource1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
streamTracerWithCustomSource1Display.DataAxesGrid.ZAxisLabels = []
streamTracerWithCustomSource1Display.DataAxesGrid.UseCustomBounds = 0
streamTracerWithCustomSource1Display.DataAxesGrid.CustomBounds = [
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    1.0,
]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
streamTracerWithCustomSource1Display.PolarAxes.Visibility = 0
streamTracerWithCustomSource1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
streamTracerWithCustomSource1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
streamTracerWithCustomSource1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
streamTracerWithCustomSource1Display.PolarAxes.CustomBounds = [
    0.0,
    1.0,
    0.0,
    1.0,
    0.0,
    1.0,
]
streamTracerWithCustomSource1Display.PolarAxes.EnableCustomRange = 0
streamTracerWithCustomSource1Display.PolarAxes.CustomRange = [0.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.RadialAxesVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.DrawRadialGridlines = 1
streamTracerWithCustomSource1Display.PolarAxes.PolarArcsVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.DrawPolarArcsGridlines = 1
streamTracerWithCustomSource1Display.PolarAxes.NumberOfRadialAxes = 0
streamTracerWithCustomSource1Display.PolarAxes.AutoSubdividePolarAxis = 1
streamTracerWithCustomSource1Display.PolarAxes.NumberOfPolarAxis = 0
streamTracerWithCustomSource1Display.PolarAxes.MinimumRadius = 0.0
streamTracerWithCustomSource1Display.PolarAxes.MinimumAngle = 0.0
streamTracerWithCustomSource1Display.PolarAxes.MaximumAngle = 90.0
streamTracerWithCustomSource1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
streamTracerWithCustomSource1Display.PolarAxes.Ratio = 1.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesColor = [
    1.0,
    1.0,
    1.0,
]
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitle = "Radial Distance"
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleLocation = "Bottom"
streamTracerWithCustomSource1Display.PolarAxes.PolarLabelVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.PolarLabelFormat = "%-#6.3g"
streamTracerWithCustomSource1Display.PolarAxes.PolarLabelExponentLocation = "Labels"
streamTracerWithCustomSource1Display.PolarAxes.RadialLabelVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.RadialLabelFormat = "%-#3.1f"
streamTracerWithCustomSource1Display.PolarAxes.RadialLabelLocation = "Bottom"
streamTracerWithCustomSource1Display.PolarAxes.RadialUnitsVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.ScreenSize = 10.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleFontFamily = "Arial"
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleFontFile = ""
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleBold = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleItalic = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleShadow = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTitleFontSize = 12
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelFontFamily = "Arial"
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelFontFile = ""
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelBold = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelItalic = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelShadow = 0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisLabelFontSize = 12
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextFontFamily = "Arial"
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextFontFile = ""
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextBold = 0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextItalic = 0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextShadow = 0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTextFontSize = 12
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = (
    "Arial"
)
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ""
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
streamTracerWithCustomSource1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
streamTracerWithCustomSource1Display.PolarAxes.EnableDistanceLOD = 1
streamTracerWithCustomSource1Display.PolarAxes.DistanceLODThreshold = 0.7
streamTracerWithCustomSource1Display.PolarAxes.EnableViewAngleLOD = 1
streamTracerWithCustomSource1Display.PolarAxes.ViewAngleLODThreshold = 0.7
streamTracerWithCustomSource1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
streamTracerWithCustomSource1Display.PolarAxes.PolarTicksVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
streamTracerWithCustomSource1Display.PolarAxes.TickLocation = "Both"
streamTracerWithCustomSource1Display.PolarAxes.AxisTickVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.AxisMinorTickVisibility = 0
streamTracerWithCustomSource1Display.PolarAxes.ArcTickVisibility = 1
streamTracerWithCustomSource1Display.PolarAxes.ArcMinorTickVisibility = 0
streamTracerWithCustomSource1Display.PolarAxes.DeltaAngleMajor = 10.0
streamTracerWithCustomSource1Display.PolarAxes.DeltaAngleMinor = 5.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
streamTracerWithCustomSource1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
streamTracerWithCustomSource1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
streamTracerWithCustomSource1Display.PolarAxes.ArcMajorTickSize = 0.0
streamTracerWithCustomSource1Display.PolarAxes.ArcTickRatioSize = 0.3
streamTracerWithCustomSource1Display.PolarAxes.ArcMajorTickThickness = 1.0
streamTracerWithCustomSource1Display.PolarAxes.ArcTickRatioThickness = 0.5
streamTracerWithCustomSource1Display.PolarAxes.Use2DMode = 0
streamTracerWithCustomSource1Display.PolarAxes.UseLogAxis = 0

# hide data in view
Hide(surfaceVectors1, renderView1)

# hide data in view
Hide(maskPoints1, renderView1)

# show color bar/color legend
streamTracerWithCustomSource1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set active source
SetActiveSource(slice1)

# show data in view
slice1Display = Show(slice1, renderView1, "GeometryRepresentation")

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# set active source
SetActiveSource(maskPoints1)

# set active source
SetActiveSource(streamTracerWithCustomSource1)

# turn off scalar coloring
ColorBy(streamTracerWithCustomSource1Display, None)

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(pLUT, renderView1)

# get color legend/bar for sLUT in view renderView1
sLUTColorBar = GetScalarBar(sLUT, renderView1)
sLUTColorBar.AutoOrient = 1
sLUTColorBar.Orientation = "Vertical"
# sLUTColorBar.WindowLocation = 'UpperRightCorner'
sLUTColorBar.Position = [0.89, 0.02]
sLUTColorBar.Title = "s"
sLUTColorBar.ComponentTitle = ""
sLUTColorBar.TitleJustification = "Centered"
sLUTColorBar.HorizontalTitle = 0
sLUTColorBar.TitleOpacity = 1.0
sLUTColorBar.TitleFontFamily = "Arial"
sLUTColorBar.TitleFontFile = ""
sLUTColorBar.TitleBold = 0
sLUTColorBar.TitleItalic = 0
sLUTColorBar.TitleShadow = 0
sLUTColorBar.TitleFontSize = 8
sLUTColorBar.LabelOpacity = 1.0
sLUTColorBar.LabelFontFamily = "Arial"
sLUTColorBar.LabelFontFile = ""
sLUTColorBar.LabelBold = 0
sLUTColorBar.LabelItalic = 0
sLUTColorBar.LabelShadow = 0
sLUTColorBar.LabelFontSize = 8
sLUTColorBar.AutomaticLabelFormat = 1
sLUTColorBar.LabelFormat = "%-#6.3g"
sLUTColorBar.DrawTickMarks = 1
sLUTColorBar.DrawTickLabels = 1
sLUTColorBar.UseCustomLabels = 0
sLUTColorBar.CustomLabels = []
sLUTColorBar.AddRangeLabels = 1
sLUTColorBar.RangeLabelFormat = "%-#6.1e"
sLUTColorBar.DrawAnnotations = 1
sLUTColorBar.AddRangeAnnotations = 0
sLUTColorBar.AutomaticAnnotations = 0
sLUTColorBar.DrawNanAnnotation = 0
sLUTColorBar.NanAnnotation = "NaN"
sLUTColorBar.TextPosition = "Ticks left/bottom, annotations right/top"
sLUTColorBar.ReverseLegend = 0
sLUTColorBar.ScalarBarThickness = 16
sLUTColorBar.ScalarBarLength = 0.33

# change scalar bar placement
# sLUTColorBar.WindowLocation = 'AnyLocation'
sLUTColorBar.Position = [0.8686695278969958, 0.1095890410958904]
sLUTColorBar.ScalarBarLength = 0.3300000000000004

# change solid color
streamTracerWithCustomSource1Display.AmbientColor = [
    0.7137254901960784,
    0.7137254901960784,
    0.7137254901960784,
]
streamTracerWithCustomSource1Display.DiffuseColor = [
    0.7137254901960784,
    0.7137254901960784,
    0.7137254901960784,
]

# Properties modified on streamTracerWithCustomSource1Display
streamTracerWithCustomSource1Display.Opacity = 0.5

# create a new 'Annotate Time'
tt = float(sys.argv[1])
annotateTime1 = AnnotateTime()
annotateTime1.Format = "Time: {time:f}"

# show data in view
annotateTime1Display = Show(annotateTime1, renderView1, "TextSourceRepresentation")

# trace defaults for the display properties.
annotateTime1Display.TextPropMode = "2D Text Widget"
annotateTime1Display.Interactivity = 1
annotateTime1Display.Opacity = 1.0
annotateTime1Display.FontFamily = "Arial"
annotateTime1Display.FontFile = ""
annotateTime1Display.Bold = 0
annotateTime1Display.Italic = 0
annotateTime1Display.Shadow = 0
annotateTime1Display.FontSize = 18
annotateTime1Display.Justification = "Left"
# annotateTime1Display.WindowLocation = 'UpperLeftCorner'
annotateTime1Display.Position = [0.05, 0.05]
annotateTime1Display.BasePosition = [0.0, 0.0, 0.0]
annotateTime1Display.TopPosition = [0.0, 1.0, 0.0]
annotateTime1Display.FlagSize = 1.0
annotateTime1Display.BillboardPosition = [0.0, 0.0, 0.0]
annotateTime1Display.DisplayOffset = [0, 0]

# update the view to ensure updated data information
renderView1.Update()

# Rescale transfer function
sLUT.RescaleTransferFunction(0, 0.15)

# Rescale transfer function
sPWF.RescaleTransferFunction(0, 0.15)

# current camera placement for renderView1
renderView1.CameraPosition = [
    -0.04555426634505638,
    0.0034725007420105284,
    0.004980766728780848,
]
renderView1.CameraFocalPoint = [
    -0.00410274657454531,
    0.010955481532701906,
    0.004939271456374059,
]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraViewAngle = 6.952789699570816
renderView1.CameraParallelScale = 0.016769382003033996

# save screenshot
SaveScreenshot(
    "test-%.3f.png" % tt,
    renderView1,
    ImageResolution=[1165, 801],
    FontScaling="Scale fonts proportionally",
    OverrideColorPalette="WhiteBackground",
    StereoMode="No change",
    TransparentBackground=0,
    # PNG options
    CompressionLevel="5",
)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [
    -0.04555426634505638,
    0.0034725007420105284,
    0.004980766728780848,
]
renderView1.CameraFocalPoint = [
    -0.00410274657454531,
    0.010955481532701906,
    0.004939271456374059,
]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraViewAngle = 6.952789699570816
renderView1.CameraParallelScale = 0.016769382003033996

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
