import Qt3D.Core 2.10
import Qt3D.Render 2.14
import Qt3D.Extras 2.14
import QtQuick.Controls 2.4 as QQC2
import QtQml 2.2

// This file was initially a copy of MainSceneFrameGraph.qml, but with Entitites such as volume rendering turned off.
// LDI is still generated so that intersection between slice and geometry is shown.

CameraSelector {
    id: cameraSelector
    property real aspectRatio: 1

    // The free slice(s) visible in this viewport. -1 means all of them, 0,1,2 means only that one slice
    property int visibleSlice: -1

    property alias sliceLayers: slicesLayerFilter.layers
    property alias objectManipulatorScaleCompensation: manipulatorScaleCompensationParameter.value
    property bool showVolumeBoundingBox: true




    NoPicking {

        /*
        // 1) Render BBox / Ground Plane
        LayerFilter {
            layers: [bboxLayer]
            RenderStateSet {
                enabled: parent.enabled
                renderStates: [
                    BlendEquationArguments {
                        sourceRgb: BlendEquationArguments.SourceAlpha
                        destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                        sourceAlpha: BlendEquationArguments.SourceAlpha
                        destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                    },
                    BlendEquation { blendFunction: BlendEquation.Add },
                    DepthTest { depthFunction: DepthTest.Less },
                    CullFace { mode: CullFace.Back }
                  // , MultiSampleAntiAliasing {}
                ]
                NoDraw {
                    enabled: !showVolumeBoundingBox
                }
            }
        }
*/

        
        // 1a) Test Entities with or without LDI
        NoDraw{
            enabled: false //!root3D.volumePreviewEnabled || _sceneManager.clipModeEnabled

            LayerFilter {
                layers: [clearViewAlignedQuadLayer]
                RenderStateSet {
                    MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                }
            }

            // Render geometry into LDI 
            LayerFilter {
                layers: [testingMenuLayer]
                enabled: true //root3D.volumePreviewEnabled
                RenderStateSet {
                    MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                    renderStates: [
                        DepthTest { depthFunction: DepthTest.Always },
                        NoDepthMask {},
                      //  MultiSampleAntiAliasing {},
                        CullFace { mode: CullFace.NoCulling },
                        BlendEquationArguments {
                            sourceRgb: BlendEquationArguments.SourceAlpha
                            destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                            sourceAlpha: BlendEquationArguments.SourceAlpha
                            destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                        },
                        BlendEquation { blendFunction: BlendEquation.Add }
                    ]
//                        RenderPassFilter {
//                            matchAny: [FilterKey { name: "pass"; value: "ldi"; }]
//                        }
                }
            }

            LayerFilter {
                layers: [sortViewAlignedQuadLayer]
                RenderStateSet {
                    MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                }
            }
        }


     // 1b)  Fixed-axis slices.  Hmm, seems to render free-axis sliders as well..

        NoDraw {
            enabled: true//_sceneManager.clipModeEnabled
            LayerFilter {
                id: slicesLayerFilter
                layers: [slicesLayer]
//                ClearBuffers {
//                    buffers: ClearBuffers.DepthBuffer

                    // Render Slices in back to front order
                    RenderStateSet {
                        MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                        renderStates: [
                            BlendEquationArguments {
                                sourceRgb: BlendEquationArguments.SourceAlpha
                                destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                                sourceAlpha: BlendEquationArguments.One
                                destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                            },
                            BlendEquation { blendFunction: BlendEquation.Add },
                            DepthTest { depthFunction: DepthTest.Less },
                            CullFace { mode: CullFace.NoCulling }
                          //  ,MultiSampleAntiAliasing {}
                        ]
//                        SortPolicy {
//                            sortTypes: [SortPolicy.BackToFront]
//                        }
                    }
               // }
            }
        }
        


        // 2) Render Volume and Annotations Volume
/*
        // Render Volume first
        RenderStateSet {
            renderStates: [
                BlendEquationArguments {
                    sourceRgb: BlendEquationArguments.SourceAlpha
                    destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                    sourceAlpha: BlendEquationArguments.One
                    destinationAlpha: BlendEquationArguments.One
                },
                BlendEquation { blendFunction: BlendEquation.Add },
                DepthTest { depthFunction: DepthTest.Less }
             //   MultiSampleAntiAliasing {}
            ]


             // Render entrybuffer into texture
            LayerFilter {
                layers : [entryLayer]
                RenderTargetSelector {target: gVolEntryBuffer
                    ClearBuffers {buffers: ClearBuffers.ColorBuffer;  clearColor : "#00000000"  //fgRoot.clearColor
                        RenderStateSet {renderStates: [CullFace { mode: CullFace.Back }]   
                           // RenderPassFilter {matchAny: [ FilterKey { name: "pass"; value: "geometry" } ]}
                        }
                    }
                }  
            } 

            LayerFilter {
                // Render exitbuffer and volume rendering
                layers: [volumeLayer]

                RenderStateSet {
                    MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                    renderStates: [CullFace { mode: CullFace.Front }]
//                    RenderPassFilter {
//                        enabled: !_preferences.checkBoxIsovalue
//                        //matchAny: FilterKey { name: "pass"; value: "ldi"; }
//                    }
//                    RenderPassFilter {
//                        enabled: _preferences.checkBoxIsovalue
//                        //matchAny: FilterKey { name: "pass"; value: "iso"; }
//                    }
                }
            }


        //////////// Render LDI geometry that is outside of the screen footprint of the volume rendering ////////////

            // Render volume-exitbuffer into texture to find the footprint of the volumerendering (not using entrybuffer as that can be clipped by nearplane)
            LayerFilter {
                layers : [entryLayer]
                RenderTargetSelector {target: gVolEntryBuffer
                    ClearBuffers {buffers: ClearBuffers.ColorBuffer;  clearColor : "#00000000"  //fgRoot.clearColor
                        RenderStateSet {renderStates: [CullFace { mode: CullFace.Front }]
                           // RenderPassFilter {matchAny: [ FilterKey { name: "pass"; value: "geometry" } ]}
                        }
                    }
                }
            }

            // Render geometry outside of volume footprint
            LayerFilter {
                layers : [geometryViewAlignedQuadLayer]
                RenderStateSet {
                    MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
                    renderStates: [
                        BlendEquationArguments {
                            sourceRgb: BlendEquationArguments.SourceAlpha
                            destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                            sourceAlpha: BlendEquationArguments.One
                            destinationAlpha: BlendEquationArguments.One
                        },
                        BlendEquation { blendFunction: BlendEquation.Add },
                        DepthTest { depthFunction: DepthTest.Less }
                     //   MultiSampleAntiAliasing {}
                    ]
                }
            }
        ////////////////////////////////////////////////


          //  ClearBuffers {buffers: ClearBuffers.DepthBuffer}
            
        }
*/


    }


    // Render Cylinder
    LayerFilter {
        layers: [cylinderLayer]

        RenderStateSet {
            renderStates: [
                // Technically correct blending parameters would be
                //                BlendEquationArguments {
                //                    sourceRgb: BlendEquationArguments.SourceAlpha
                //                    destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                //                    sourceAlpha: BlendEquationArguments.SourceAlpha
                //                    destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                //                }
                // but in practice when dealing with transparency and volume rendering
                // it makes it impossible to recognize the cylinder
                BlendEquationArguments {
                    sourceRgb: BlendEquationArguments.SourceAlpha
                    destinationRgb: BlendEquationArguments.OneMinusSourceAlpha
                    sourceAlpha: BlendEquationArguments.SourceAlpha
                    destinationAlpha: BlendEquationArguments.OneMinusSourceAlpha
                },
                BlendEquation { blendFunction: BlendEquation.Add },
                CullFace { mode: CullFace.Back }
             //  , MultiSampleAntiAliasing {}
            ]
        }

    }

    // 4) Seed Point
    LayerFilter {
        layers: [seedPointLayer]
        RenderStateSet {
            renderStates: [
            //    MultiSampleAntiAliasing {},
                NoDepthMask {}
            ]
        }
    }

    // 5) PointedLines
    NoPicking {
        LayerFilter {
            layers: [pointedLinesLayer]
            RenderStateSet {
                renderStates: [
                    NoDepthMask {}
                   //, MultiSampleAntiAliasing {}
                ]
            }
        }

        // 6) Axis Gnomon
        LayerFilter {
            layers: [axisGnomonLayer]
            Viewport {
                id: axesGnomonViewport
                readonly property real viewportHeight: 0.1 * root3D.aspectRatio * aspectRatio
                normalizedRect: Qt.rect(0.0, 1.0 - viewportHeight, 0.1, viewportHeight)
                RenderStateSet {
                    renderStates: [
                        NoDepthMask {}
                    //  ,  MultiSampleAntiAliasing {}
                    ]
                    // No depth write and no depth testing
                }
            }
        }
    }


    // 9) Manipulators
    LayerFilter {
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
/*
    // 10) Sculpting
    LayerFilter {
        layers: [sculptingPreviewerLayer]
        ClearBuffers {
            buffers: ClearBuffers.DepthBuffer
            RenderStateSet {
                MemoryBarrier { waitFor: MemoryBarrier.ShaderImageAccess }
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
            }
        }
    }
    */
}
