import QtQuick 2.14
import QtQuick.Shapes 1.14

Rectangle
{
	id:root
	
	property int posX:100
	property int posY:100
	property string message:" message 1"
	property int sizePolicy:14
	property color col:"white"
	property bool gras:false
	property bool italique:false
	property string police:"Helvetica"
	
	
	objectName : "tooltip3d"
	color: "transparent" //"#66222222"
    visible:true
    x: root.posX-20
    y: root.posY-height-15
    width:controlText.paintedWidth+10
    height:controlText.paintedHeight+10
    border.color: "#FF7F00"
	border.width: 0
	radius: 4
	
	
	
	Shape 
	{
	    //width: 200
	    //height: 150
	   // anchors.centerIn: parent
	    anchors.fill: parent
	    ShapePath {
	        strokeWidth: 2
	        strokeColor:"#FF7F00"
	        //strokeStyle: ShapePath.DashLine
	        startX: 0; startY: 0
	        fillColor:"#77000000"
	        capStyle:ShapePath.RoundCap
	        PathPolyline
			{
				
				id:ppl
				path: [Qt.point(20.0,root.height+15.0), Qt.point(15.0,root.height),Qt.point(0.0,root.height),Qt.point(0.0,0),
				Qt.point(root.width,0),Qt.point(root.width,root.height),Qt.point(25.0,root.height),Qt.point(20.0,root.height+15.0)]
				
				
			
			}
	     
	    }
	}
	
	
	Text {
	        id: controlText
	        anchors.fill: parent
	        horizontalAlignment: Text.AlignHCenter
	        verticalAlignment: Text.AlignVCenter
	        font.family:root.police
	        font.pixelSize: root.sizePolicy
	        font.bold: root.gras
	        font.italic:root.italique
	        color: root.col
	        text: root.message
	    }
}