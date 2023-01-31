import QtQuick 2.0
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Scene3D 2.3
import Murat 1.0

Page
{
 	Scene3D {
        id: scene3D
        anchors.fill: parent
        focus: true
        multisample: true
        aspects: ["input", "logic"]
        hoverEnabled: true
        View3D {
        
            id: view3D
            viewportRect: Qt.rect(scene3D.x, scene3D.y, scene3D.width, scene3D.height)
            backgroundColor:"#FF9797"// Material.color(Material.Grey, Material.Shade800)
            //volumePreviewEnabled: volumePreviewButton.checked
           // annotationsPreviewEnabled: annotationsPreviewButton.checked
        }
    }
    
    header: ToolBar {
        id: toolBar
        Flow {
            anchors.fill: parent
            ToolButton {
                id: showCylinderPropButton
                checkable: true
                checked:true
                text: "Layout 3D split"
                icon.source: "qrc:/slicer/icons/fontawesome/crosshairs-solid.svg"
                onCheckedChanged: {
                view3D.changeSplit3D(checked);
                //	ViewportManager.modeSplit3D = checked
                   // if (checked) {
                    	//showSplit3D();
                        //showSlicePropButton.checked = false
                        //showFreeSlicePropButton.checked = false
                    //}
                }
            }
          /*  ToolButton {
                id: showSlicePropButton
                checkable: true
                text: "Slices"
                icon.source: "images/fontawesome/sliders-h.svg"
                onCheckedChanged: {
                    if (checked) {
                        showCylinderPropButton.checked = false
                        showFreeSlicePropButton.checked = false
                    }
                }
            }
            ToolButton {
                id: showFreeSlicePropButton
                checkable: true
                text: "Free Slices"
                icon.source: "images/fontawesome/sliders-h.svg"
                onCheckedChanged: {
                    if (checked) {
                        showCylinderPropButton.checked = false
                        showSlicePropButton.checked = false
                    }
                }
            }
            ToolSeparator {}
            ToolButton {
                id: showCameraPropButton
                checkable: true
                text: "Camera"
                icon.source: "images/fontawesome/video.svg"
            }
            ToolButton {
                id: volumePreviewButton
                checkable: true
                text: "Volume"
                icon.source: "images/fontawesome/eye.svg"
            }
            ToolButton {
                id: annotationsPreviewButton
                checkable: true
                text: "Annotations"
                icon.source: "images/fontawesome/eye.svg"
            }
            ToolSeparator {}
            ButtonGroup { id: manipulatorsButtonGroup }
            ToolButton {
                id: globalManipulators
                checkable: true
                checked: view3D.editMode === ObjectManipulator.Global
                text: checked ? "Global" : "Local"
                onToggled: {
                    if (view3D.editMode === ObjectManipulator.Global)
                        view3D.editMode = ObjectManipulator.Local
                    else
                        view3D.editMode = ObjectManipulator.Global
                }
                icon.source: "images/fontawesome/globe.svg"
            }
            ToolSeparator {}
            ToolButton {
                id: testingMenuButton
                checkable: true
                text: "Testing Menu"
                icon.source: "images/fontawesome/globe.svg"
            }
            ToolSeparator {}*/
            ToolButton {
                text: "+X"
                onClicked: mainRoot.lookAtX()
            }
            ToolButton {
                text: "-X"
                onClicked: mainRoot.lookAtXNeg()
            }
            ToolButton {
                text: "+Y"
                onClicked: mainRoot.lookAtY()
            }
            ToolButton {
                text: "-Y"
                onClicked: mainRoot.lookAtYNeg()
            }
            ToolButton {
                text: "+Z"
                onClicked: mainRoot.lookAtZ()
            }
            ToolButton {
                text: "-Z"
                onClicked: mainRoot.lookAtZNeg()
            }
            ToolButton {
                text: "Last Perspective"
                onClicked: view3D.restoreCustomView()
            }
            ToolButton {
                text: "Reset"
                onClicked: view3D.resetMainCamera()
            }
           /* ToolSeparator {}
            ToolButton {
                checkable: true
                text: "Manipulator"
                icon.source: "images/fontawesome/eye-slash-solid.svg"
                checked: view3D.manipulatorHidden
                onToggled: view3D.manipulatorHidden = !view3D.manipulatorHidden
            }
            ToolSeparator {}
            ToolButton {
                id: preferencesButton
                checkable: true
                text: ""
                icon.source: "images/fontawesome/cog.svg"
            }*/
        }
    }
    
    footer: ToolBar {
        id: statusBar
        height: statusBarRow.implicitHeight
        focusPolicy: Qt.NoFocus
        RowLayout {
            id: statusBarRow
            anchors.fill: parent

            Label {
                id: dimensionsLabel
                text: " message" //"Dimensions I(%1), J(%2), K(%3)".arg(_sceneManager.dimensions.x) .arg(_sceneManager.dimensions.y).arg(_sceneManager.dimensions.z)
                Layout.alignment: Qt.AlignVCenter
            }
           /* Item { Layout.fillWidth: true }
            ProgressBar {
                indeterminate: true
                visible: _sceneManager.isLoading
                Layout.alignment: Qt.AlignVCenter
            }*/
        }
    }
}