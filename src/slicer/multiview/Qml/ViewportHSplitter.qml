import QtQuick 

Item {
    id: root
    property real originalPosX: 0.5
    property real originalPosY: 0.5
    property var widthX:0
    property SubViewport viewParent
    signal posChanged(real pos);

    //parent: viewParent
    anchors.fill: parent

    Item {
    	id: item
        
        width:parent.width
        height:6
        anchors {
            left: parent.left
            right: parent.right
        }
          Rectangle
        {
       		 anchors.fill: parent
       		 color:"#FF00FF"
        }
        MouseArea {
            anchors.fill: parent
            cursorShape: Qt.SplitVCursor
            onPositionChanged: {
                var sceneY = mapToItem(scene3D, mouseX, mouseY).y
                posChanged(sceneY / scene3D.height)
            }
        }
    }
    
    Component.onCompleted:{
    
    	console.log(" scene3D.width :"+ scene3D.width+ " ," +item.x +" , "+ item.y+ " viewport H spliiter : "+item.width+" , height"+ item.height );
    }
}
