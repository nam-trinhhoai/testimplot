import Qt3D.Core 2.10
import Qt3D.Render 2.10
import Qt3D.Extras 2.13

Entity {
    property Layer layer
    
    
   

    readonly property GeometryRenderer mesh:  GeometryRenderer {
        id: lineRenderer
        primitiveType: GeometryRenderer.Lines
        Buffer {
            id: vertexBuffer
          //  type: Buffer.VertexBuffer
            data: {
                var orig = Qt.vector3d(0, 0, 0)
                var x = Qt.vector3d(1,0, 0)
                var y = Qt.vector3d(0, 1, 0)
                var z = Qt.vector3d(0, 0, 1)
                
           
                var vertices = [orig, x, x, x,
                                orig, y, y, y,
                                orig, z, z, z]

                var vertexArray = new Float32Array(vertices.length * 3)
                var i = 0;
                vertices.forEach(function(vec3) {
                    vertexArray[i++] = vec3.x;
                    vertexArray[i++] = vec3.y;
                    vertexArray[i++] = vec3.z;
                });
                return vertexArray
            }
        }
        geometry: Geometry {
            attributes: [
                Attribute {
                    name: defaultPositionAttributeName
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    byteOffset: 0
                    byteStride: 6 * 4 // 6 * sizeof(float)
                    count: 6
                    buffer: vertexBuffer
                },
                Attribute {
                    name: defaultColorAttributeName
                    attributeType: Attribute.VertexAttribute
                    vertexBaseType: Attribute.Float
                    vertexSize: 3
                    byteOffset: 3 * 4 // 3 * sizeof(float)
                    byteStride: 6 * 4 // 6 * sizeof(float)
                    count: 6
                    buffer: vertexBuffer
                }
            ]
        }
    }

    readonly property Material material: Material {
        effect: Effect {
            techniques: Technique {
                graphicsApiFilter {
                    api: GraphicsApiFilter.OpenGL
                    majorVersion: 3
                    minorVersion: 2
                    profile: GraphicsApiFilter.CoreProfile
                }

                renderPasses: RenderPass {
                    shaderProgram: ShaderProgram {
                        vertexShaderCode: loadSource("qrc:/slicer/multiview/Qml/Shaders/axes_gnomon.vert")
                        fragmentShaderCode: loadSource("qrc:/slicer/multiview/Qml/Shaders/axes_gnomon.frag")
                    }
                }
            }
        }
    }
    
     Transform{
     		id:trGlobal
            translation: Qt.vector3d(-0.0, 0.0, 0.0)
            scale: 0.1
        }

    components: [
    	trGlobal,
        lineRenderer,
        material,
        layer
    ]


    Effect {
        id: textEffect
        techniques: Technique {
            graphicsApiFilter {
                api: GraphicsApiFilter.OpenGL
                majorVersion: 3
                minorVersion: 2
                profile: GraphicsApiFilter.CoreProfile
            }

            renderPasses: RenderPass {
                shaderProgram: ShaderProgram {
                    vertexShaderCode: loadSource("qrc:/slicer/multiview/Qml/Shaders/gnomon-text.vert")
                    fragmentShaderCode: loadSource("qrc:/slicer/multiview/Qml/Shaders/gnomon-text.frag")
                }
            }
        }
    }

    Entity {
        readonly property GeometryRenderer renderer: ExtrudedTextMesh {
            depth: 0.1
            font.family: "Roboto"
            text: "X"
        }

        readonly property Transform transform: Transform {
            translation: Qt.vector3d(0.6, 0.0, 0.0)
            scale: 0.4
        }

        readonly property Material material:  Material {
            effect: textEffect
            parameters: Parameter { name: "color"; value: "red"; }
        }

        components: [material, renderer, transform]
    }

    Entity {
        readonly property GeometryRenderer renderer: ExtrudedTextMesh {
            depth: 0.1
            font.family: "Roboto"
            text: "Y"
        }

        readonly property Transform transform: Transform {
            translation: Qt.vector3d(0.0, 0.6, 0.0)
            scale: 0.4
        }

        readonly property Material material:  Material {
            effect: textEffect
            parameters: Parameter { name: "color"; value: "green"; }
        }

        components: [material, renderer, transform]
    }

    Entity {
        readonly property GeometryRenderer renderer: ExtrudedTextMesh {
            depth: 0.1
            font.family: "Roboto"
            text: "Z"
        }

        readonly property Transform transform: Transform {
            translation: Qt.vector3d(0.0, 0.0, 0.6)
            scale: 0.4
        }

        readonly property Material material:  Material {
            effect: textEffect
            parameters: Parameter { name: "color"; value: "blue"; }
        }

        components: [material, renderer, transform]
    }
}
