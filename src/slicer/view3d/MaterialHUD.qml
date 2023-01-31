import Qt3D.Core 2.14
import Qt3D.Render 2.14

Material {
	id: root
	
	property color maincolor: Qt.rgba(0.0, 0.0, 0.0, 1.0)	
	property alias diffuse: diffuseTextureImage.source
	property double aspectRatio: 1.0
	property double bottom: 0.0
	property double left: 0.0
	property double right: 100.0
	property double top: 100.0
	property double screenWidth:400.0
	
	parameters: [
		Parameter {
			name: "maincolor"
			value: Qt.vector3d(root.maincolor.r, root.maincolor.g, root.maincolor.b)
		},
		Parameter {
			name: "aspectRatio"
			value: aspectRatio
		},
		Parameter {
			name: "bottom"
			value: bottom
		},
		Parameter {
			name: "left"
			value: left
		},
		Parameter {
			name: "right"
			value: right
		},
		Parameter {
			name: "top"
			value: top
		},
		Parameter {
			name: "screenWidth"
			value: screenWidth
		},
		Parameter {
			name: "diffuseTexture"
			value: Texture2D {
				id: diffuseTexture
				minificationFilter: Texture.LinearMipMapLinear
				magnificationFilter: Texture.Linear
				wrapMode {
					x: WrapMode.Repeat
					y: WrapMode.Repeat
				}
				generateMipMaps: true
				maximumAnisotropy: 16.0
				TextureImage { id: diffuseTextureImage }
			}
		}
	]


	effect: Effect {
		property string vertex: "qrc:/shaders/qt3d/materials/hud_phong.vert"
       	property string fragment: "qrc:/shaders/qt3d/materials/hud_phong.frag"
	
	
		FilterKey {
			id: forward
			name: "renderingStyle"
			value: "forward"
		}
		ShaderProgram {
			id: gl3Shader
			vertexShaderCode: loadSource(parent.vertex)
			fragmentShaderCode: loadSource(parent.fragment)
		}
		
		
		techniques: [
			// OpenGL 3.1
			Technique {
				filterKeys: [forward]
				graphicsApiFilter {
					api: GraphicsApiFilter.OpenGL
					profile: GraphicsApiFilter.CoreProfile
					majorVersion: 3
					minorVersion: 1
				}
				renderPasses: RenderPass {
					shaderProgram: gl3Shader
				}
			}
		]
	}
}
   
   
