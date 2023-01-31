import QtQuick 2.10

Item {
    id: root
    property real originalPos: 0.5
    signal posChanged(real pos);

    parent: scene3D
    anchors.fill: parent

    Item {
        x: originalPos * parent.width - 3
        width: 6
        anchors {
            top: parent.top
            bottom: parent.bottom
        }
        /*Rectangle
        {
       		 anchors.fill: parent
       		 color:"#FF00FF"
        }*/
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SplitHCursor
            onPositionChanged: {
                var sceneX = mapToItem(scene3D, mouseX, mouseY).x
                posChanged(sceneX / scene3D.width)
            }
        }
    }
}
