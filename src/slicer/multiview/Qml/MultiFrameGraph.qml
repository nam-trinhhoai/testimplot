import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Input 2.13
import QtQml 2.10
import QtQuick 2.13 as QQ2
import QtQuick.Controls 2.4 as QQC2
import QtQuick.Controls.Material 2.1 as QQCM
import Murat 1.0

RenderSurfaceSelector {
    id: fgRoot
    property SubViewport currentViewport: mainViewport
    readonly property Entity currentCamera: currentViewport.camera
    readonly property Entity mainCamera: mainViewport.camera
    property alias clearColor: clearBuffers.clearColor
    readonly property bool is3DViewportActive: currentViewport === mainViewport

    ViewportManager {
        id: vpManager
        mainViewport: mainViewport
        freeSliceSubviewPortHeight: 0.7
        freeSliceSubviewPortWidth: 0.4
        currentViewMode: {
        return ViewportManager.FreeSliceIntersectionMode
           /* if (_sceneManager.cylinderEnabled)
                return ViewportManager.View2DIntersectionMode
            if (_sceneManager.freeSlicesEnabled)
                return ViewportManager.FreeSliceIntersectionMode*/
           // return ViewportManager.FreeViewMode
        }
    }

    // Texture Extraction (Cylinder/FreeSlice)
  /*  LayerFilter {
        // Launch Compute job and retrieve buffer data
        DispatchCompute {
            BufferCapture { enabled:false// _sceneManager.waitingForUpdatedCylinderSkin
                                    // FreeSlice Extraction Disabled
                                   
            }
        }
        layers: [volumeLayer]
    }*/

    ClearBuffers {
        id: clearBuffers
        buffers: ClearBuffers.ColorDepthBuffer
        NoDraw {}
    }

    // Handle to control size of viewport in FreeSlice mode
    ViewportSplitter {
        enabled: vpManager.currentViewMode === ViewportManager.FreeSliceIntersectionMode
        visible: enabled
        originalPos: 1.0 - vpManager.freeSliceSubviewPortWidth
        onPosChanged: vpManager.freeSliceSubviewPortWidth = Math.max(0.35, 1.0 - pos)
    }
    
    

    // Main 3D Viewport
    NoDraw {
        enabled: !mainViewport.enabled
        SubViewport {
            id: mainViewport
            //toolbarVisible: true
            active: currentViewport === this
            onSelected: currentViewport = this
            // Perspective Projection
            camera: Camera{//StereoCamera {
                aspectRatio: mainViewport.aspectRatio
                position: root3D.dimensions
                upVector: Qt.vector3d(0.0, 1.0, 0.0)
                viewCenter: Qt.vector3d(0.0, 0.0, 0.0)
                nearPlane: Math.max(1, position.length() - (root3D.maxDimension * 4))
                farPlane: position.length() + (root3D.maxDimension * 4)
            }
            name: "3D View"

            manager: vpManager
            QQ2.Component.onCompleted: vpManager.addViewport(mainViewport, ViewportManager.PlaceHolder_1)
            canBeMaximized: vpManager.currentViewMode === ViewportManager.FreeSliceIntersectionMode

          /*  StereoRenderingFrameGraph {
                viewportRect: mainViewport.normalizedRect
                stereoCamera: mainViewport.camera
                aspectRatio: mainViewport.aspectRatio
                clearColor: fgRoot.clearColor
                sliceLayers: [
                    slicesLayer,
                    freeSlice1Layer,
                    freeSlice2Layer,
                    freeSlice3Layer
                ]
                objectManipulatorScaleCompensation: controller.mode === CameraController.Mode2D ? (controller.view2DZoomFactor / controller.view3DZoomFactor) : 1.0
            }*/
        }
    }
  
    /*
    // Main 3D Viewport
    NoDraw {
        enabled: !mainViewport2.enabled
        SubViewport {
            id: mainViewport2
            active: currentViewport === this
            onSelected: currentViewport = this
            // Perspective Projection
            camera: Camera{//StereoCamera {
                aspectRatio: mainViewport2.aspectRatio
                position: root3D.dimensions
                upVector: Qt.vector3d(0.0, 1.0, 0.0)
                viewCenter: Qt.vector3d(0.0, 0.0, 0.0)
                nearPlane: Math.max(1, position.length() - (root3D.maxDimension * 4))
                farPlane: position.length() + (root3D.maxDimension * 4)
            }
            name: "3D View"

            manager: vpManager
            QQ2.Component.onCompleted: vpManager.addViewport(mainViewport2, ViewportManager.PlaceHolder_1)
            canBeMaximized: vpManager.currentViewMode === ViewportManager.FreeSliceIntersectionMode
            
          

        }
    }
    */

    MouseDevice { id: mouseDevice }

    // FreeSlice Viewport
    NodeInstantiator {
        id: repeater
        model: 2

        NoDraw {
            enabled: !freeSliceViewport.enabled
            
            
            
            
           
            SubViewport {
                id: freeSliceViewport
                active: currentViewport === this
                onSelected: currentViewport = this
                isSliceViewport: true
                canBeMaximized: true
                
            /*    ViewportHSplitter {
    
		    	viewParent: freeSliceViewport
		        enabled:true// vpManager.currentViewMode === ViewportManager.FreeSliceIntersectionMode
		        visible: enabled
		        originalPosX: 1.0 - vpManager.freeSliceSubviewPortWidth
		        originalPosY: 1.0 - vpManager.freeSliceSubviewPortHeight
		        widthX: vpManager.freeSliceSubviewPortWidth
		        onPosChanged:
		        {
		        	console.log(" on poschanged :"+pos)
		         vpManager.freeSliceSubviewPortHeight = Math.max(0.2,  pos)
		         }
		    }*/
              

                headerBarColor: {
                    if (model.index === 0)
                        return QQCM.Material.color(QQCM.Material.Red)
                    if (model.index === 1)
                        return QQCM.Material.color(QQCM.Material.Green)
                    return QQCM.Material.color(QQCM.Material.Blue)
                }

                property bool flipView: false

                camera: Camera {
                    /*readonly property vector3d viewAxis: {
                        // The mainSlice axis is +Y (as by default Qt3D planes point toward +Y)
                        // Since we want to display in X, Y, Z order
                        // X normal is firstOrthogonalAxis
                        // Y normal is mainAxis
                        // Z normal is secondOrthogonalAxis
                        if (model.index === 0)
                            return _sceneManager.freeSliceInfo.firstOrthogonalSlice.axis
                        if (model.index === 1)
                            return _sceneManager.freeSliceInfo.mainSlice.axis

                        freeSliceViewport.flipView = true
                        return _sceneManager.freeSliceInfo.secondOrthogonalSlice.axis
                    }*/
                    position:Qt.vector3d(0,-5000,0)// _sceneManager.seedPointPosition.plus(viewAxis.times(-root3D.maxDimension).times(freeSliceViewport.flipView ? -1.0 : 1.0))
                    viewCenter:Qt.vector3d(0,0,0)// _sceneManager.seedPointPosition
                    upVector:
                    {
                    	return Qt.vector3d(0.0, 0.0, -1.0)
                      /*if (_sceneManager.isViewModeSeismic)
                      {
                         if (model.index === 0) return Qt.vector3d(0.0, 0.0, -1.0)
                          return Qt.vector3d(-1.0, 0.0, 0.0)
                      }

                      return   _sceneManager.freeSliceInfo.upVectorForViewAxis(viewAxis)*/
                    }
                    projectionType: CameraLens.OrthographicProjection
                    nearPlane: -position.length() - (root3D.maxDimension * 4)
                    farPlane: position.length() + (root3D.maxDimension * 4)
                    right: (root3D.maxDimension * root3D.aspectRatio * freeSliceViewport.aspectRatio * 0.75) / 1.0//controller.view2DZoomFactor
                    top: (root3D.maxDimension * 0.75) / 1.0//controller.view2DZoomFactor
                    left: -right
                    bottom: -top
                }
                readonly property var names: ["Section", "Map", "Z"]
                name: "Slice " + freeSliceViewport.names[model.index]

                manager: vpManager
                QQ2.Component.onCompleted: vpManager.addViewport(freeSliceViewport, ViewportManager.PlaceHolder_2 + model.index)

                headerBarComponent: FreeSliceViewportHeaderBar {
                    index: model.index
                    onFlip:  freeSliceViewport.flipView = !freeSliceViewport.flipView
                    viewAxis: Qt.vector3d(0,1,0) //freeSliceViewport.camera.viewAxis
                   
                    onRecenter: _sceneManager.seedPointPositionChanged()
                }
                   
                
              
                SliceSceneFrameGraph {
                    camera: freeSliceViewport.camera
                    // Disable CameraSelector when parent isn't enabled
                    // to prevent picking in undesired areas
                    enabled: parent.enabled
                    aspectRatio: freeSliceViewport.aspectRatio
                   // objectManipulatorScaleCompensation: controller.mode === CameraController.Mode3D ? (controller.view3DZoomFactor / controller.view2DZoomFactor) : 1.0

                    visibleSlice: model.index
                    sliceLayers: freeSlice2Layer
                    /* (_sceneManager.isViewModeSeismic & _sceneManager.timeHorizons.isLoaded & _preferences.checkBoxIsovalue) ?
                                        model.index === 0
                                         ? freeSliceComputeLayer
                                         : model.index === 1
                                           ? freeSlice2Layer
                                           : freeSlice3Layer
                                    :
                                        model.index === 0
                                         ? freeSlice1Layer
                                         : model.index === 1
                                           ? freeSlice2Layer
                                           : freeSlice3Layer
*/

                    showVolumeBoundingBox: false
                    
                    
                    
               
//                    Connections {
//                        target: _sceneManager
//                        onViewModeSeismicChanged: {
//                            sliceLayers: model.index === 0
//                                           ? freeSliceComputeLayer
//                                           : model.index === 1
//                                             ? freeSlice2Layer
//                                             : freeSlice3Layer
//                            console.log("Seismic slice mode changed: " + model.index)
//                        }
//                    }

//                    Connections {
//                        target: _sceneManager
//                        onViewModeStandardChanged: {
//                            sliceLayers: model.index === 0
//                                           ? freeSlice1Layer
//                                           : model.index === 1
//                                             ? freeSlice2Layer
//                                             : freeSlice3Layer
//                            console.log("Standard slice mode changed: " + model.index)
//                        }
//                    }

//                    Connections {
//                        target: _sceneManager
//                        onViewModeMicroCTChanged: {
//                            sliceLayers: model.index === 0
//                                           ? freeSlice1Layer
//                                           : model.index === 1
//                                             ? freeSlice2Layer
//                                             : freeSlice3Layer

//                            console.log("MicroCT slice mode changed: " + model.index)
//                        }
//                    }
                }


                // Mouse Wheel Handler to rotate FreeSlices
                MouseHandler {
                    sourceDevice: mouseDevice
                    enabled: freeSliceViewport.active

                    readonly property vector3d rotationAxis: Qt.vector3d(0,1,0) //freeSliceViewport.camera.viewAxis

                    onWheel: {
                        // TO DO: Qt3D doesn't seem to handle the enabled property on MouseHandler
                        if (enabled && wheel.modifiers & Qt.ControlModifier) {
                            var speed = wheel.modifiers & Qt.ShiftModifier ? 5.0 : 1.0
                            var baseRotation = _sceneManager.freeSliceTransform.rotation
                            var newRotation = _sceneManager.freeSliceTransform.fromAxisAndAngle(rotationAxis, speed * (wheel.angleDelta.y < 0 ? -1.0 : 1.0))
                            _sceneManager.freeSliceTransform.rotation = MathUtils.multQuaternions(newRotation, baseRotation)
                        }
                    }
                }
            }
            
            
            
            
        }
    }
    /*
    ViewportHSplitter {
    
    	viewParent: freeSliceViewport
        enabled:true// vpManager.currentViewMode === ViewportManager.FreeSliceIntersectionMode
        visible: enabled
        originalPosX: 1.0 - vpManager.freeSliceSubviewPortWidth
        originalPosY: 1.0 - vpManager.freeSliceSubviewPortHeight
        widthX: vpManager.freeSliceSubviewPortWidth
        onPosChanged:
        {
        	console.log(" on poschanged :"+pos)
         vpManager.freeSliceSubviewPortHeight = Math.max(0.2,  pos)
         }
    }*/

}
