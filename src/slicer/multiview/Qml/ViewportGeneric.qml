import Qt3D.Core 2.10
import Qt3D.Render 2.10
import QtQuick 2.12 as QQ2
import QtQuick.Controls 2.4 as QQC2
import QtQuick.Controls.Material 2.1 as QQCM
import QtQuick.Layouts 1.3
import QtQuick
import Murat 1.0


 NoDraw {
 	id: root
      enabled: !viewportGen.enabled
       //property SubViewport parentView:null
        property string name
         property ViewportManager manager: null
         property bool canBeMaximized: false
         property bool toolbarVisible: false
         property bool active: false
    property Entity camera: null
    property Viewport currViewport:viewportGen
    property alias headerBarComponent: headerBarLoader.sourceComponent
    property color headerBarColor: Qt.darker(QQCM.Material.color(QQCM.Material.Yellow), 1.2)
   
    signal selected()
    readonly property real aspectRatio: normalizedRect.width / normalizedRect.height
    property bool isSliceViewport: false
   
    readonly property bool isMaximized: canBeMaximized && manager.maximizedViewport === root

Viewport {
    id: viewportGen
    
     
    
   
    
    


	QQ2.Component {
        id: tabButton
        QQC2.TabButton { }
    }
    // QtQuick Viewport window frame
    readonly property QQ2.Item frame: QQ2.Rectangle {
        visible: root.enabled
        enabled: visible
        parent: scene3D
        x: normalizedRect.x * scene3D.width
        y: normalizedRect.y * scene3D.height
        width: normalizedRect.width * scene3D.width
        height: normalizedRect.height * scene3D.height
        color: "transparent"
        
          
        
        border {
            width: 2
            color: (root.active && scene3D.focus) ? QQCM.Material.color(QQCM.Material.LightBlue) : "grey"
        }
        QQ2.MouseArea {
            anchors.fill: parent
            enabled: !scene3D.focus || !active
            acceptedButtons: Qt.AllButtons

            onPressed: {
                scene3D.focus = true;
                root.selected();
                mouse.accepted = false;
            }

            onWheel: {
                scene3D.focus = true;
                root.selected();
                wheel.accepted = false;
            }
        }

        QQ2.DropArea {
            id: dropArea
            // Hold a reference to the view to make swapping
            // possible
            readonly property var view: root
            anchors.fill: parent
            enabled: !root.active
            QQ2.Rectangle {
                anchors.fill: parent
                color: draggableViewport.color
                opacity: draggableViewport.opacity
                visible: dropArea.containsDrag
            }
        }

        QQ2.Rectangle {
            id: draggableViewport
            width: frame.width
            height:frame.height
            visible: headerBarMa.drag.active && QQ2.Drag.target === null
            color: QQCM.Material.color(QQCM.Material.LightBlue)
            opacity: 0.5
            QQ2.Drag.active: headerBarMa.drag.active
            QQ2.Drag.hotSpot.x: width * 0.5
            QQ2.Drag.hotSpot.y: 0
            states: QQ2.State {
                when: headerBarMa.drag.active
                QQ2.ParentChange { target: draggableViewport; parent: scene3D }
                QQ2.AnchorChanges { target: draggableViewport; anchors.verticalCenter: undefined; anchors.horizontalCenter: undefined }
            }
        }

        QQ2.Rectangle {
            id: headerBar
            color: Qt.darker(root.headerBarColor, root.active ? 1 : 1.4)
            anchors {
                top: parent.top
                left: parent.left
                right: parent.right
                margins: 2
            }
            height: nameLabel.implicitHeight + 15 //Math.max(nameLabel.implicitHeight + 15, headerBarLoader.implicitHeight)
            QQC2.Label {
                id: nameLabel
                text: name
                anchors {
                    //left: parent.left
                    horizontalCenter:parent.horizontalCenter
                    verticalCenter: parent.verticalCenter
                    leftMargin: 10
                }
            }
            QQ2.MouseArea {
                id: headerBarMa
                anchors.fill: parent
                drag.target: draggableViewport

                onPressed: {
                    scene3D.focus = true
                    root.selected()
                }

                onReleased: {
                    // Do we need to swap views?
                    if (draggableViewport.QQ2.Drag.target !== null) {
                        // Swapping basically mean switch the viewport rects
                        manager.swapViewports(root, draggableViewport.QQ2.Drag.target.view)
                    }

                    // Reset draggableViewport x/y for next time
                    draggableViewport.x = 0
                    draggableViewport.y = 0
                    draggableViewport.parent = frame
                }
            }
          /*  FreeSliceViewportHeaderBar {
                    index: 0  //model.index
                    onFlip:  freeSliceViewport.flipView = !freeSliceViewport.flipView
                    viewAxis: Qt.vector3d(0,1,0) //freeSliceViewport.camera.viewAxis
                   
                   // onRecenter: _sceneManager.seedPointPositionChanged()
                }
            
            */
            
          /*  QQ2.Loader {
                id: headerBarLoader
                anchors {
                    left: name.right
                    right: canBeMaximized ? maximizeButton.left : parent.right
                }
            }*/
           
            QQC2.ToolButton {
                id: addButton
                visible: true
                enabled: visible
                onClicked:
                {
                
                console.log("--> root3D.modeSplit3 : "+root3D.modeSplit3)
                	manager.nbView3D =manager.nbView3D+1
                	var component = Qt.createComponent("SubViewport.qml");
                	var nom = "3DView "+manager.nbView3D;
    				var viewport =component.createObject(fgRoot, {name:nom,manager: manager, canBeMaximized: true, parentView: false });
          
                	manager.addViewportMulti(0,viewport,ViewportManager.PlaceHolder_2+manager.nbView3D,root3D.modeSplit3)// manager.maximizedViewport = (isMaximized ? null : root)
                	
                	//ajout d'un onglet
                	var val = tabbar.count+1
                	var tab = tabButton.createObject(tabbar, {text: "View " + val})
            		tabbar.addItem(tab)
                	
                }
                icon.source: "qrc:/slicer/icons/add.png"
                anchors {
                    verticalCenter: parent.verticalCenter
                    left: parent.left
                }
            }
            RowLayout {
             anchors.fill: parent
             spacing:2
             //layoutDirection: "RightToLeft"
             Item {
       				Layout.fillWidth: true
   			 }
            QQC2.ToolButton {
                id: maximizeButton
                visible: canBeMaximized
                enabled: visible
                onClicked: manager.maximizedViewport = (isMaximized ? null : root)
                icon.source: isMaximized ? "qrc:/slicer/icons/fontawesome/minimize.svg" : "qrc:/slicer/icons/fontawesome/maximize.svg"
                Layout.alignment: Qt.AlignRight
                /*anchors {
                    verticalCenter: parent.verticalCenter
                    right: parent.right
                }*/
            }
            QQC2.ToolButton {
                id: closeButton
                visible: true
                enabled: visible
                onClicked: manager.removeViewport(this) // = (isMaximized ? null : root)
                icon.source: "qrc:/slicer/icons/close.png"
                Layout.alignment: Qt.AlignRight
                /*anchors {
                    verticalCenter: parent.verticalCenter
                    right: parent.right
                }*/
            }
            }
        }
        
       QQC2.TabBar{
        id:tabbar
        y:30
       // visible:manager.modeSplit3D
        visible:toolbarVisible && root3D.modeSplit3
	   	width:parent.width
	   	QQC2.TabButton
		{
			enabled:true
			text:"View 1"
		}  
	/*	QQC2.TabButton
		{
			enabled:true
			text:"View 2"
		}   
		QQC2.TabButton
		{
			enabled:false
			text:"View 3"
		}
		QQC2.TabButton
		{
			enabled:false
			visible:false
			text:"View 4"
		}   */   
	   }
     
    }
}



}
