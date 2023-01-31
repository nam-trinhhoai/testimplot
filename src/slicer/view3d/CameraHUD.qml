import QtQml 2.2

import Qt3D.Core 2.14
import Qt3D.Render 2.14
import Qt3D.Input 2.14
import Qt3D.Logic 2.14
import QtQuick.Scene3D 2.14
import Qt3D.Extras 2.14

import QtQuick.Controls 1.4
import QtQuick.Layouts  1.14
import QtQuick.Controls.Styles 1.4
import QtQuick 2.14 
import RGTSeismicSlicer 1.0

Item{
 	id: rootItem
 	
 	Rectangle { width: 100; height: 100;color: "#00B000" }
 	
	Scene3D {
	    id: scene3d
	    objectName : "scene3D"
	    anchors.fill: parent
	    anchors.margins: 10
	    focus: true
	    aspects: ["input", "logic"]
	    hoverEnabled: true
	    cameraAspectRatioMode: Scene3D.AutomaticAspectRatio
	//}
	
	//Scene3DView {
	
	//	onXChanged: console.log(("X (%1): %2").arg(model.index).arg(x))
	//	scene3D: scene3d
	//	anchors.fill: parent
	
	    Entity {
	        id: sceneRoot
	        objectName : "sceneRoot"
	
	        function viewAll() {
			camera.viewAll()
		}
			    
	        Camera {
			id: camera
			projectionType: CameraLens.PerspectiveProjection
			position: Qt.vector3d(0, 0, 20)
			viewCenter: Qt.vector3d(0, 0, 0)
			upVector: Qt.vector3d( 0.0, 1.0, 0.0 )
			fieldOfView: 45
			nearPlane : 0.1
			farPlane : 1000.0
		}
		
		SOrbitCameraController {
	               objectName : "controler"
		        camera: camera
		        linearSpeed:100
		        lookSpeed: 50
		}
		
		//FirstPersonCameraController {
		  //  camera: camera
		    //linearSpeed: 10
		    //lookSpeed: 50
		//}
	        
      	
		RenderSettings {
			id : renderSettings
			activeFrameGraph: ForwardRenderer {
				camera: camera
				clearColor: "#ffffff"
			}
		}
		
		// Event Source will be set by the Qt3DQuickWindow
		InputSettings { id: inputSettings }
	       
		components: [renderSettings, inputSettings]
		
		PhongMaterial {
			id: material
		}
		
		Texture2D{
			id: texture
			TextureImage {
			source: "qrc:/assets/cube.jpg"
			}
		}
		
		DiffuseMapMaterial {
			id: diffuseMaterial
			diffuse:  texture
			specular: Qt.rgba( 0.2, 0.2, 0.2, 1.0 )
			shininess: 2.0
		}
		
		MaterialHUD {
			id: materialHUD
			maincolor: Qt.rgba(1.0, 0.0, 0.0, 1.0)
			diffuse: "qrc:/assets/cube.jpg"
		}
		
		/*
		Entity {
			components: [
				DirectionalLight {
					intensity: 0.9
					worldDirection: Qt.vector3d(0.2, 0.6, 0.2)
				}
			]
		}*/
		
		SphereMesh {
			id: sphereMesh
			radius: 3
		}
		PlaneMesh {
			id: planeMesh
			width: 1.0
			height: 1.0
			meshResolution: Qt.size(2, 2)
		}
		
		Mesh {
			id: cameraGizmoMesh
			source: "qrc:/assets/cube.obj"
		}
		
		Transform {
			id: transformSphere
			translation: Qt.vector3d(0, 0, 5)
		}
		
		Transform {
			id: transform
			//translation: Qt.vector3d(0, 0, 5)
			rotationX: 45
		}
		
		Transform {
			id: transformHUD
			translation: Qt.vector3d(0, 0, 5)
		}
		
		Entity {
			id: cameraGizmoEntity
			objectName: "cameraGizmoEntity"
			components: [ cameraGizmoMesh, materialHUD, transformHUD ]
		}

		//Entity {
		//	id: sphereEntity
		//	objectName: "sphereEntity"
		//	components: [ sphereMesh, material, transformSphere ]
		//}		
				
		Entity {
			id: mainEntity
			objectName: "mainEntity"
			components: [ planeMesh, material, transform ]
		}		
	    }
	}
}
   
   
