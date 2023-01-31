import Qt3D.Core 2.14
import Qt3D.Render 2.14


TechniqueFilter {
    // Expose camera to allow user to choose which camera to use for rendering
    property alias camera: cameraSelector.camera
    property alias clearColor: clearBuffer.clearColor
    property alias viewportRect: viewport.normalizedRect
    property alias window: surfaceSelector.surface
    property alias externalRenderTargetSize: surfaceSelector.externalRenderTargetSize
    property alias frustumCulling: frustumCulling.enabled

    // Select the forward rendering Technique of any used Effect
    matchAll: [ FilterKey { name: "renderingStyle"; value: "forward" } ]

    RenderSurfaceSelector {
        id: surfaceSelector

        // Use the whole viewport
        SortPolicy{
			sortTypes:[
				SortPolicy.Material
			]
	
		}
        Viewport {
            id: viewport
            normalizedRect: Qt.rect(0.0, 0.0, 1.0, 1.0)

            // Use the specified camera
            CameraSelector {
                id : cameraSelector
                FrustumCulling {
                    id: frustumCulling
                    ClearBuffers {
                        id: clearBuffer
                        clearColor: "white"
                        buffers : ClearBuffers.ColorDepthBuffer
                       // NoDraw{}
                    }
                }
           
            }
            
				
            LayerFilter
		    {
		       //objectName: "opaqueFilter"
		     //  id: opaqueFilter
		       layers: [opaqueLayer]
		       
		     /*  SortPolicy{
					sortTypes:[
						SortPolicy.Material	
					]
			
				}*/
		    }
		 
		    
             LayerFilter
		    {
		    //  id: transparentFilter
		     // objectName: "transparentFilter"
		      layers: [transparentLayer] //transparentLayer]
		     //  filterMode: LayerFilter.DiscardAllMatchingLayers
		      // NoDepthMask {}
		     
		      
		     /* SortPolicy{
					sortTypes:[
						SortPolicy.BackToFront
					]
			
				}*/
		    }
            
		
		   
        }
    }
}




/*LayerFilter {
        layers: [manipulatorLayers]
        ClearBuffers {
            buffers: ClearBuffers.DepthBuffer
            RenderPassFilter {
                parameters: Parameter {
                    id: manipulatorScaleCompensationParameter
                    name: "scaleCompensation"
                    value: 1.0
                }
                RenderStateSet {
                    renderStates: [
                        DepthTest { depthFunction: DepthTest.Less },
                    //    MultiSampleAntiAliasing {},
                        BlendEquationArguments {
                            sourceRgb: BlendEquationArguments.SourceAlpha
                            destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                            sourceAlpha: BlendEquationArguments.SourceAlpha
                            destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                        },
                        BlendEquation { blendFunction: BlendEquation.Add }
                    ]
                    SortPolicy {
                        sortTypes: [SortPolicy.BackToFront]
                    }
                }
            }
        }
    }
    
    */