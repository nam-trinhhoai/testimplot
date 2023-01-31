import QtQml// 2.2

import Qt3D.Core// 2.14
import Qt3D.Render //2.14
import Qt3D.Input
import Qt3D.Logic //2.14
import QtQuick.Scene3D //2.14
import Qt3D.Extras //2.14

import QtQuick.Controls //2.14
import QtQuick.Layouts  //1.14
//import QtQuick.Controls.Styles 1.4
import QtQuick// 2.14 
import RGTSeismicSlicer 1.0

//CMAKE_LIBRARY_PATH == /usr/lib64/nvidia

Item{
 	id: rootItem
 	property var tabTooltip:[]
 	property double _zScale: 100
    property rect viewportRect
    readonly property rect glCorrectViewportRect: Qt.rect(rootItem.viewportRect.x + frameGraph.viewportRect.x * rootItem.viewportRect.width,
                                                          (1.0 - frameGraph.viewportRect.y - frameGraph.viewportRect.height) * rootItem.viewportRect.height - rootItem.viewportRect.y,
                                                          rootItem.viewportRect.width * frameGraph.viewportRect.width,
                                                          rootItem.viewportRect.height * frameGraph.viewportRect.height)
	
	
	
	
	
	
 	/*Shortcut{
        sequence :" "
        onActivated:{
            if(dashboard.opened)
            	dashboard.close();
            else
            	dashboard.open();
        }
    }*/
    
    Connections{
    	target:viewQt
    	
    	function onRemoveAllTooltip()
    	{
    		removeTooltip();
    	}
    	
    	function onAddTooltip(positionX, positionY,message,size,police,gras,italique,colour)
    	{
    		
    		addTooltip3D(positionX,positionY,message,size,police,gras, italique,colour);
    	}
    	
    	function onUpdateTooltip(index,posx,posy)
    	{
    		
    		updateTooltip3D(index,posx,posy);
    	}
    	
    	function onSendDestroyTooltip(name)
    	{
    		removeTooltip3D(name);
    	}
    	 function onSendUpdateTooltip(index, size)
	    { 	
	    	tabTooltip[index].sizePolicy = size;
	    }
	    
	    function onSendColorTooltip(index,color)
	    {
	   		tabTooltip[index].col = color;
	    }
	    
	    function onSendFontTooltip(index,policy,size,italic,bold)
	    {
	   		tabTooltip[index].police = policy;
	   		tabTooltip[index].sizePolicy = size;
	   		tabTooltip[index].gras = bold;
	   		tabTooltip[index].italique = italic;
	    }
	    
    
   
    }
    
    function addTooltip3D(posiX,posiY,texte,size,policy,bold,italic,colour)
    {
  
    	var component = Qt.createComponent("Tooltip3d.qml");
    	var tooltip = component.createObject(rootItem,{posX:posiX,posY:posiY,message:texte,sizePolicy:size,police:policy,gras:bold,italique:italic,col:colour})
    	tabTooltip.push(tooltip);
    }
    
    
    function updateTooltip3D(index, posiX,posiY)
    {
   	 	
    	tabTooltip[index].posX = posiX;
    	tabTooltip[index].posY = posiY;
    }
    
   
    
    function removeTooltip()
    {
    	for(var i=0;i<tabTooltip.length;i++)
    	{
    		tabTooltip[i].destroy();
    	}
    	tabTooltip.splice(0,tabTooltip.length);
    }
    
     
    function removeTooltip3D(name)
    {
    	var index = findTooltip(name)
    	
    	if(index >=0)
    	{
    		
    		tabTooltip[index].destroy();
    		tabTooltip.splice(index,1);
    	}
    } 
    
    function findTooltip(name)
    {
    	for(var i=0;i<tabTooltip.length;i++)
    	{
	    	if(name === tabTooltip[i].message)
	    	{
	    		return i;
	    	}
    	}
    	return -1;
    }
     
	Scene3D {
	    id: scene3d
	    objectName : "scene3D"
	    anchors.fill: parent
	    anchors.margins: 10
	    focus: true
	    aspects: ["input", "logic"]
	    hoverEnabled: true
	    cameraAspectRatioMode: Scene3D.AutomaticAspectRatio
	    multisample: true

		
		
	    Entity 
	    {
	        id: sceneRoot2
	        objectName : "sceneRoot2"

			readonly property double cameraZ: camera.position.z
			
		    function viewAll() {
		        camera.viewAll()
		    }

		    
		    Transform {
               id: transfoGlobal
               objectName : "transfoGlobal"
               scale3D: Qt.vector3d(1,1,1)
             }
             
            /* 
              Layer { 
              	id: opaqueLayer 
              	recursive: true
              	objectName:"opaqueLayer"
              	
              }
    		  Layer { 
    		  	id: transparentLayer
    		  	recursive: true
    		  	objectName:"transparentLayer"
    		 // 	NoDepthMask {}
    		  	 }
             
            */
			  
	        Camera {
	            id: camera
	            objectName : "camera"
	            projectionType: CameraLens.PerspectiveProjection
	            fieldOfView: 45
	            nearPlane : 1
	            farPlane : 100000
                    viewCenter: Qt.vector3d( 0.0, 0.0, 0.0 )
	            position: Qt.vector3d( 0.0, 0.0, -40.0 )
	            upVector: Qt.vector3d( 0.0, 1.0, 0.0 )
	            
	         
	                  
	        }
		
			CameraController {
			id: cameraController
	  	       	objectName : "controler"
	  	       	activeCamera: camera
	  	       	zoomSpeed: 4.0
	  	       	panSpeed: 0.1
	  	       	windowSize: Qt.size(scene3d.width, scene3d.height)
	  	       	viewportRect: rootItem.viewportRect  //width
	  	       	
	  	       	onFpsChanged: {
	        		textfps.text= "FPS:"+fps;
	        	}
	        	onAltitudeChanged: {
	        		var displayAltitude = altitude;
	        		if (qt3DRessource.sectionTypeQML == QMLEnumWrappers.DEPTH)
	        		{
	        			var inUnit = MtLengthUnitMETRE
	        			var outUnit = qt3DRessource.depthLengthUnitQML
	        			displayAltitude = outUnit.convertFrom(inUnit, altitude)
	        		}
	        		textaltitude.text= "Altitude:"+displayAltitude;
	        	}
	        	onSpeedCamChanged: {
	        		textspeed.text= "Speed:"+speedCam;
	        	}
	        	function cleanTextAltitude()
	        	{
	        		textaltitude.text= "Altitude:0";
	        	}
			}
			
			Entity {
			    id: sceneRoot
	            objectName : "sceneRoot"
				
			 	Transform {
					id: transfoFilsGlobal
					objectName : "transfoFilsGlobal"
					scale3D: Qt.vector3d(1,1,1)
				}
				components: [transfoFilsGlobal]		
			}
		    
		    
	     RenderSettings {
				id : renderSettings
				activeFrameGraph: ForwardRenderer 
				{
					id: frameGraph
					camera: camera
					//frustumCulling: false		
					clearColor: "#FF19232D"	
				//	SortPolicy{
				//	sortTypes:[
					//	SortPolicy.FrontToBack	
				//	]//FrontToBack //BackToFront
			
				//	}
				}
				//renderPolicy:RenderSettings.OnDemand
				pickingSettings.pickMethod: PickingSettings.TrianglePicking
				pickingSettings.faceOrientationPickingMode: PickingSettings.FrontAndBackFace
			
			}
		
			// Event Source will be set by the Qt3DQuickWindow
			InputSettings { id: inputSetting}
		    components: [renderSettings, inputSetting,transfoGlobal] 
		    
		    
		  /*  components: [
		        RenderSettings {
		            id: renderSettings
		            activeFrameGraph: ForwardRenderer {
		            	id: frameGraph
		                camera: camera
		                clearColor: "transparent"
		                //showDebugOverlay: true
		            }
		            pickingSettings.pickMethod: PickingSettings.PrimitivePicking
					pickingSettings.faceOrientationPickingMode: PickingSettings.FrontAndBackFace  //FrontAndBackFace
					pickingSettings.pickResultMode: PickingSettings.NearestPick
		        },
		        InputSettings { },
		        
		    ]
		    */
		    
			Texture2D{
				id: texture
				TextureImage {
				source: "qrc:/assets/arrows.png"
				}
			}
		
			MaterialHUD {
				id: materialHUD
				maincolor: Qt.rgba(255.0/70.0, 255.0/81.0, 255.0/107.0, 0.0)
				bottom: camera.bottom
				left: camera.left
				right: camera.right
				top: camera.top
				screenWidth:rootItem.width
				aspectRatio: camera.aspectRatio
				diffuse: "qrc:/assets/cube3.png"
				
			
			}
	
				
			Mesh {
				id: cameraGizmoMesh
				source: "qrc:/assets/cube3.obj"
			}
			
			Transform {
				id: transformHUD
				translation: Qt.vector3d(0, 0, 500) 
			}
			
			Entity {
				id: cameraGizmoEntity
				objectName: "cameraGizmoEntity"
				components: [ cameraGizmoMesh, materialHUD, transformHUD ]
			}
			
			
	    }
	}
	

	// curseur par defaut
	MouseArea{
		id:mousearea
		anchors.fill: parent
		acceptedButtons: Qt.NoButton
		cursorShape:Qt.ArrowCursor
	}

	

	

    
    Rectangle
    {
    	id:linetooltip
    	objectName : "linetooltip"
    	visible:false
    	width:10
	    height:2
	    border.color: "#FF7F00"
		border.width: 2
		radius: 1
		x:0
	    y:0
	    rotation: 0
	    transformOrigin:Item.Left
    	
    }
    
    
    Rectangle{
		id:recttooltip
		objectName : "recttooltip"
		color: "#66222222"
	       visible:false
	       anchors.centerIn: parent
	       width:300
	       height:200
	       border.color: "#FF7F00"
		   border.width: 2
		   radius: 10
	       
		//transform: Rotation { origin.x: 0; origin.y: 0; axis { x: 0; y: 1; z: 0 } angle: 30}
	
		 Label
		 {
		    id:nametooltip   
		    objectName : "nametooltip"
	    	x:20
	    	y:10
	    	color:"#FFFFFF"
	    	text:""
		 }
		 Button
	    {
	    	//id:deselecttooltip
		    //objectName : "deselecttooltip"
			x:parent.width -30
			y:5
			width:25
			height:25
			

			Text {
		        id: controlText
		        anchors.fill: parent
		        horizontalAlignment: Text.AlignHCenter
		        verticalAlignment: Text.AlignVCenter
		        font.pixelSize: 24
		        color: "#FFFFFF"
		        text: "X"
		    }
			background: Rectangle {
		      //  implicitWidth: 100
		      //  implicitHeight: 40
		      //  opacity: enabled ? 1 : 0.3
		      //  border.color: control.down ? "#17a81a" : "#21be2b"
		      //  border.width: 1
		     //   radius: 2
        		color: "transparent"  // I update background color by this
    		}
			onClicked:{
				viewQt.deselectWell();
				
			}
		}
		
		Label{
		     id:statustooltip
		     objectName : "statustooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:uwitooltip
		     objectName : "uwitooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:datetooltip
		     objectName : "datetooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:domaintooltip
		     objectName : "domaintooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:elevtooltip
		     objectName : "elevtooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:datumtooltip
		     objectName : "datumtooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:velocitytooltip
		     objectName : "velocitytooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Label{
		     id:ihstooltip
		     objectName : "ihstooltip"
		     x:20
		     y:30
		     color:"#FFFFFF"
		     text:""
	    }
	    Button
	    {
	    	id:deselecttooltip
		    objectName : "deselecttooltip"
			anchors.horizontalCenter: parent.horizontalCenter
			y:60
			width:120
			height:25
			text:"deselect"
			onClicked:{
				viewQt.hideWell();
			}
		}
	}
	
   
    Label{
        id:"tooltip"
    	objectName : "tooltip"
    	x:50
    	y:100
    	color:"#FFFFFF"
    	text:""
    }
	
	Rectangle{
		id:infos3d
		objectName : "infos3d"
		anchors.right: parent.right
		anchors.top: parent.top
		width:82
		height:60
		//color: "#253545"
		 border.color: "#FF7F00"
		 border.width: 1
		 color: "#66222222"
		 radius:2
	//	border.color:"#354555"
	//	border.width:2
	ColumnLayout{
		spacing:0
		anchors.right: parent.right
		Layout.alignment: Qt.AlignRight
		
		Rectangle
		{
			 Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
	      	color: "transparent"
	        Layout.preferredWidth: 80
	        Layout.preferredHeight: 20
	        Layout.leftMargin:3
	        Text { 
	        	width:parent.width
	        	id:textfps
	        	 anchors.verticalCenter: parent.verticalCenter
	        	 font.pointSize: 8
	        	text: "FPS:"+rootItem._fps ; font.bold: false;color: "white"; 
	        }
	    }
	    Rectangle
		{
			 Layout.alignment:Qt.AlignRight | Qt.AlignVCenter
	        color: "transparent"
	        Layout.preferredWidth: 80
	        Layout.preferredHeight: 20
	       Layout.leftMargin:3
	        Text { 
	        	id:textspeed
	        	 font.pointSize: 8
	        	  anchors.verticalCenter: parent.verticalCenter
	        	text: "Speed:"+0 ; font.bold: false;color: "white"; 
	        }
	        }
	        Rectangle
		{
			 Layout.alignment: Qt.AlignRight | Qt.AlignVCenter
	        //color: "#253545"
	        color: "transparent"
	        Layout.preferredWidth: 80
	        Layout.preferredHeight: 20
	        Layout.leftMargin:3
	        Text { 
	        	id:textaltitude
	        	 font.pointSize: 8
	        	  anchors.verticalCenter: parent.verticalCenter
	        	text: "Altitude:"+0 ; font.bold: false;color: "white"; 
	        }
	        }
	
	}
	}

	Qt3DRessource {
		id: qt3DRessource
		objectName : "qt3DRessource"
		onSectionTypeQMLChanged: {
			cameraController.cleanTextAltitude()
		}
		onDepthLengthUnitQMLChanged: {
			cameraController.cleanTextAltitude()
		}
	}
	
	/*
	ColumnLayout{
		anchors.right: parent.right
		Layout.preferredWidth: 100
	     Layout.maximumWidth: 100
		
		RowLayout {
				 
	        id: fpsLayout
	        Layout.preferredWidth: 100
	         Layout.maximumWidth: 100
	        anchors.left: parent.left
	        anchors.leftMargin: parent.width -150
	        anchors.right: parent.right
	        anchors.rightMargin: 0
	        //anchors.top: scene3d.top-120
	        spacing: 5
	        
	        Text { 
	        	width:parent.width
	        	id:textfps
	        	text: "FPS:"+rootItem._fps ; font.bold: true;color: "gray"; 
	        }
		} 
		
		RowLayout {
	        id: speedLayout
	        Layout.preferredWidth: 100
	        Layout.maximumWidth: 100
	        anchors.left: parent.left
	        anchors.leftMargin: parent.width -150
	        anchors.right: parent.right
	        anchors.rightMargin: 0
	       // anchors.top: scene3d.top-200
	        spacing: 5
	        Text { 
	        	id:textspeed
	        	text: "Speed:"+0 ; font.bold: true;color: "gray"; 
	        }
		} 
		RowLayout {
	        id: altitudeLayout
	        Layout.preferredWidth: 100
	         Layout.maximumWidth: 100
	        anchors.left: parent.left
	        anchors.leftMargin: parent.width -150
	        anchors.right: parent.right
	        anchors.rightMargin: 0
	       // anchors.top: scene3d.top-200
	        spacing: 5
	        Text { 
	        	id:textaltitude
	        	text: "Altitude:"+0 ; font.bold: true;color: "gray"; 
	        }
		} 
	}*/
	
	Dashboard
	{
		id:dashboard
	}
	
	
/*	Drawer {
        id: drawer
        x :rootItem.width*0.1
        width: rootItem.width*0.8
        height: 100
        edge:Qt.BottomEdge
        visible:false
        closePolicy:Popup.CloseOnEscape
        modal:false
        
         Label
        {
	        text:"contenu du drawer"
	        anchors.centerIn: parent
        }
        Overlay.modal:Rectangle
        {
        	color:"transparent"
        }
        
        background:Rectangle
        {
        	border.color:"#FF7F00"
        	border.width: 1
        	radius:6
        	color: "#88222222"
        	opacity:1
        }
        
        
         Label
        {
	        text:"contenu du drawer"
	        anchors.centerIn: parent
	        color:"white"
        }
        
        
       
    }*/
	
}   
