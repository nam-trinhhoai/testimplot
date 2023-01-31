import Qt3D.Core 2.14
import Qt3D.Render 2.14
import Qt3D.Input 2.14
import Murat 1.0


Entity {
    id: root3D
     readonly property vector3d dimensions: Qt.vector3d(10.0,10.0,10.0)//_sceneManager.dimensions
     property color backgroundColor: "#999797"
  		property bool modeSplit3: true
     property rect viewportRect
       readonly property rect glCorrectViewportRect: Qt.rect(root3D.viewportRect.x + frameGraph.currentViewport.normalizedRect.x * root3D.viewportRect.width,
                                                          (1.0 - frameGraph.currentViewport.normalizedRect.y - frameGraph.currentViewport.normalizedRect.height) * root3D.viewportRect.height - root3D.viewportRect.y,
                                                          root3D.viewportRect.width * frameGraph.currentViewport.normalizedRect.width,
                                                          root3D.viewportRect.height * frameGraph.currentViewport.normalizedRect.height)
  
     
         Layer { id: volumeLayer }
    Layer { id: volumeIsoPrecomputeLayer }
    Layer { id: wellTrajectoriesLayer }
    Layer { id: clearViewAlignedQuadLayer}
    Layer { id: sortViewAlignedQuadLayer}
    Layer { id: geometryViewAlignedQuadLayer}
    Layer { id: entryLayer}
    Layer { id: geotimeSliceZLayer}
    Layer { id: bboxLayer }
    Layer { id: cylinderLayer }
    Layer { id: axisGnomonLayer; recursive: true }
    Layer { id: slicesLayer; recursive: true }
    Layer { id: freeSliceComputeLayer; recursive: true }
    Layer { id: freeSlice1Layer; recursive: true }
    Layer { id: freeSlice2Layer; recursive: true }
    Layer { id: freeSlice3Layer; recursive: true }
    Layer { id: manipulatorLayers; recursive: true }
    Layer { id: testingMenuLayer; recursive: true }
    //Layer { id: testingMenuLayer2; recursive: true }
    Layer { id: pointedLinesLayer }
    Layer { id: seedPointLayer; recursive: true }
    Layer { id: sculptingPreviewerLayer; recursive: true }
     
     
     
     function changeSplit3D(checked)
     {
     	console.log(" split 3D"+checked)
     	root3D.modeSplit3 = checked
     }


     CameraController {
        id: controller
        activeCamera: frameGraph.currentCamera
        enabled:true/* !isPicking &&
                 !currentManipulator.isActive &&
                 !sculptManager.isSculpting*/
        windowSize: Qt.size(scene3D.width, scene3D.height)
        view3DCamera: frameGraph.mainCamera
        view3DTarget:null// root3D.manipulatorTarget !== volumetricCube.freeSlice ? root3D.manipulatorTarget : null
        zoomSpeed: 4.0
        panSpeed: 4.0
        mode: frameGraph.is3DViewportActive ? CameraController.Mode3D : CameraController.Mode2D
        viewportRect: root3D.glCorrectViewportRect
        // TO DO: We will likely need to adjust actions/behavior
        // based on the type of the view currently active and the mode
        // we are in (FreeSlice/View2D)
    }
    
    AxesGnomon {
        layer: axisGnomonLayer
    }
    
     
     
     components: [
        RenderSettings {
            activeFrameGraph: MultiFrameGraph {
                id: frameGraph
                clearColor: root3D.backgroundColor
            }
            renderPolicy: RenderSettings.Always
            pickingSettings {
                pickResultMode: PickingSettings.NearestPriorityPick
                faceOrientationPickingMode: PickingSettings.FrontAndBackFace
                pickMethod: PickingSettings.TrianglePicking
            }
        },
        InputSettings {}
    ]
    
    
    
}