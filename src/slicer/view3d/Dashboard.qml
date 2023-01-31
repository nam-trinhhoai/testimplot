import QtQuick.Controls 2.14
import QtQuick 2.14
import QtQuick.Layouts 1.14
//import QtQuick.Extras 1.4
//import QtQuick.Controls.Styles 1.4

Drawer {
        id: drawer
        x :rootItem.width*0.1
    
        width: rootItem.width*0.8
        height: 100
        edge:Qt.BottomEdge
        visible:false
        closePolicy:Popup.CloseOnEscape
        modal:false
        
     
     
       
        Overlay.modal:Rectangle
        {
        	color:"transparent"
        }
        
        background:Rectangle
        {
        	border.color:"#FF7F00"
        	border.width: 1
        	radius:6
        	color: "#66000000"
        	opacity:1
        }
         Rectangle {

            property var rowSpan: 7
            property var columSpan: 3

            Layout.column: 0
            Layout.row: 1

            Layout.preferredWidth: (parent.width / parent.columns) * columSpan
            Layout.preferredHeight: (parent.height / parent.rows) * rowSpan

            Layout.columnSpan: columSpan
            Layout.rowSpan: rowSpan

            color: "red"

        }
        
        
        GridLayout
        {
       	 	anchors.fill:parent
        	//spacing:0
        	//anchors.margins:10
        	columns:10
        	rows: 4
        	Layout.margins:0
        	columnSpacing:0
        	rowSpacing:0
        	
        	Rectangle {

	            property var rowSpan: 1
	            property var columSpan: 2
	
	            Layout.column: 0
	            Layout.row: 0
	            
	           
	
	            Layout.preferredWidth:120// (parent.width / parent.columns) * columSpan
	            Layout.preferredHeight:parent.height*0.5 // (parent.height / parent.rows) * rowSpan
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
	            color: "transparent"
	            
            	Image
	        	{
	        		// Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
	        		x:20
	        		width:60
	        		height:60
	        		source:"qrc:/slicer/icons/camera_blanc.png"
	        	}

       		}	
       		Rectangle {

	            property var rowSpan: 1
	            property var columSpan: 2
	
	            Layout.column: 0
	            Layout.row: 2
	
	            Layout.preferredWidth:120// (parent.width / parent.columns) * columSpan
	            Layout.preferredHeight:parent.height*0.5 // (parent.height / parent.rows) * rowSpan
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
	            color: "transparent"
	            
            	CheckBox {
			        checked: true
			        Text {
			        x: 40
			        y: 5
				        text: "Attach cam\n to ortho"
				        color: "white"
				    }
			      
			    }

       		}	
       		Rectangle {
				id:rect1
	            property var rowSpan: 4
	            property var columSpan: 2
	
	            Layout.column: 2
	            Layout.row: 0
	
	            Layout.preferredWidth:((parent.width -120)/ 8) * columSpan
	            Layout.preferredHeight:parent.height // (parent.height / parent.rows) * rowSpan
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
				//Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
	            color: "transparent"
	            MouseArea {
			        anchors.fill: parent
			        hoverEnabled:true
			        onWheel: {
			           // console.log("==>"+wheel.angleDelta.y);
			            control.currentIndex += wheel.angleDelta.y/120.0;
			        }
			       
			       }
			        
			        
			       Tumbler {
					    id: control
					    //model: 10
					    model: [3, 2,1,0,-1,-2,-3]
					    wrap:false
					    currentIndex:3
					   
					   height:parent.height
					
					    background: Item {
					    	Rectangle {
					            opacity: 0.2
					            border.color: "#000000"
					            width: parent.width
					            height: parent.height
					            anchors.bottom: parent.bottom
					        }
					  /*      Rectangle {
					            opacity: control.enabled ? 0.8 : 0.1
					            border.color: "#000000"
					            width: parent.width
					            height: 1
					            anchors.top: parent.top
					        }
					
					        Rectangle {
					            opacity: control.enabled ? 0.8 : 0.1
					            border.color: "#000000"
					            width: parent.width
					            height: 1
					            anchors.bottom: parent.bottom
					        }
					        */
					       
					    }
					
					    delegate: Text {
					    	
					        text: modelData //qsTr("Item %1").arg(modelData + 1)
					        font: control.font
					        color:"white"
					        horizontalAlignment: Text.AlignHCenter
					        verticalAlignment: Text.AlignVCenter
					        opacity: 1.0 - Math.abs(Tumbler.displacement) / (control.visibleItemCount / 2)
					    }
					
					    Rectangle {
					        anchors.horizontalCenter: control.horizontalCenter
					        y: control.height * 0.4
					        width: 40
					        height: 1
					        color: "#21be2b"
					    }
					
					    Rectangle {
					        anchors.horizontalCenter: control.horizontalCenter
					        y: control.height * 0.6
					        width: 40
					        height: 1
					        color: "#21be2b"
    }
}
	        /*    Dial
	        	{
	        		anchors.fill:parent
	        		id:dial1
	        	}
	        	Text{
	        		//anchors.fill:parent
	        		width:100
	        		height:40
	        		x:(rect1.width- 50)*0.5
	        		y: (rect1.height- 20)*0.5
	        		text:"Speed"
	        	}*/

       		}
       		Rectangle {
				id:rect2
	            property var rowSpan: 4
	            property var columSpan: 2
	
	            Layout.column: 4
	            Layout.row: 0
	
	            Layout.preferredWidth:((parent.width -120)/ 8) * columSpan
	            Layout.preferredHeight:parent.height
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
	            color: "transparent"
	            MouseArea {
			        anchors.fill: parent
			        onWheel: {
			           // console.log("==>"+wheel.angleDelta.y);
			            dial2.value += wheel.angleDelta.y/2400.0;
			        }
			        }
	            Dial
	        	{
	        		anchors.fill:parent
	        		id:dial2
	        	}
	        	Text{
	        		//anchors.fill:parent
	        		width:100
	        		height:40
	        		x:(rect2.width- 50)*0.5
	        		y: (rect2.height- 20)*0.5
	        		text:"Altitude"
	        	}

       		}
       		Rectangle {
				id:rect3
	            property var rowSpan: 4
	            property var columSpan: 2
	
	            Layout.column: 6
	            Layout.row: 0
	
	            Layout.preferredWidth:((parent.width -120)/ 8) * columSpan
	            Layout.preferredHeight:parent.height
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
	            color: "transparent"
	            MouseArea {
			        anchors.fill: parent
			        onWheel: {
			           // console.log("==>"+wheel.angleDelta.y);
			            dial3.value += wheel.angleDelta.y/2400.0;
			        }
			        }
	            Dial
	        	{
	        		anchors.fill:parent
	        		id:dial3
	        	}
	        	Text{
	        		//anchors.fill:parent
	        		width:100
	        		height:40
	        		x:(rect3.width- 50)*0.5
	        		y: (rect3.height- 20)*0.5
	        		text:"Distance"
	        	}

       		}
       		Rectangle {
				id:rect4
	            property var rowSpan: 4
	            property var columSpan: 2
	
	            Layout.column: 8
	            Layout.row: 0
	
	            Layout.preferredWidth:((parent.width -120)/ 8) * columSpan
	            Layout.preferredHeight:parent.height-20
	
	            Layout.columnSpan: columSpan
	            Layout.rowSpan: rowSpan
	
	            color: "transparent"
	            
	            MouseArea {
			        anchors.fill: parent
			        onWheel: {
			           // console.log("==>"+wheel.angleDelta.y);
			            dial4.value += wheel.angleDelta.y/2400.0;
			        }
			        }
	            Dial
	        	{
	        		anchors.fill:parent
	        		id:dial4
	        		/* background: Rectangle {
					        x: dial4.width / 2 - width / 2
					        y: dial4.height / 2 - height / 2
					        width: Math.max(64, Math.min(dial4.width, dial4.height))
					        height: width
					        color: "transparent" //"transparent"
					        radius: width / 2
					        border.color: "#FF7F00" //dial4.pressed ? "#17a81a" : "#21be2b"
					        opacity: dial4.enabled ? 1 : 0.3
					    }*/
					    
					     handle: Rectangle {
						        id: handleItem
						        x: dial4.background.x + dial4.background.width / 2 - width / 2
						        y: dial4.background.y + dial4.background.height / 2 - height / 2
						        width: 16
						        height: 16
						        color: dial4.pressed ? "#FF7F00" : "#CC7F00"   //dial4.pressed ? "#17a81a" : "#21be2b"
						        radius: 8
						        antialiasing: true
						        opacity: dial4.enabled ? 1 : 0.3
						        transform: [
						            Translate {
						                y: -Math.min(dial4.background.width, dial4.background.height) * 0.4 + handleItem.height / 2
						            },
						            Rotation {
						                angle: dial4.angle
						                origin.x: handleItem.width / 2
						                origin.y: handleItem.height / 2
						            }
						        ]
						    }
	        		
	        	/*	style: DialStyle {
	        				 tickmark: Rectangle {
                   				//CircularButtonStyleHelper {
                   				color:"blue"
                   				///}

              			  }
                        //labelInset: outerRadius * 0
                    }*/
	        		
	        	}
	        	Text{
	        		//anchors.fill:parent
	        		width:100
	        		height:40
	        		x:(dial4.width-20)*0.5
	        		y: (dial4.height- 20)*0.5
	        		text:"Tilt"
	        	}

       		}
        
	 
        }
        
     /*   RowLayout
        {
        	anchors.fill:parent
        	spacing:10
        	anchors.margins:10
        	
        	Gauge
        	{
	        	Layout.fillHeight:true
	        	minimumValue:-40
	        	maximumValue:40
	        	tickmarkStepSize:20
	        	
	        }
	        
	        
        	CircularGauge{
        		id:speedGauge
        		Layout.fillHeight:true
        	
				        		
        	}
        	
        	Dial
        	{
        		id:dial1
        		//Layout.fillWidth:true
        		Layout.fillHeight:true
        		
        		 MouseArea {
			        anchors.fill: parent
			        onWheel: {
			           // console.log("==>"+wheel.angleDelta.y);
			            dial1.value += wheel.angleDelta.y/2400.0;
			        }

     
    }
        	}
        	Dial
        	{
        		id:dial2
        		//Layout.fillWidth:true
        		Layout.fillHeight:true
        	}
        	*/
        	
        	
  /*     Dial {
    id: control
    background: Rectangle {
        x: control.width / 2 - width / 2
        y: control.height / 2 - height / 2
        width: Math.max(64, Math.min(control.width, control.height))
        height: width
        color: "transparent"
        radius: width / 2
        border.color: control.pressed ? "#17a81a" : "#21be2b"
        opacity: control.enabled ? 1 : 0.3
    }

    handle: Rectangle {
        id: handleItem
        x: control.background.x + control.background.width / 2 - width / 2
        y: control.background.y + control.background.height / 2 - height / 2
        width: 16
        height: 16
        color: control.pressed ? "#17a81a" : "#21be2b"
        radius: 8
        antialiasing: true
        opacity: control.enabled ? 1 : 0.3
        transform: [
            Translate {
                y: -Math.min(control.background.width, control.background.height) * 0.4 + handleItem.height / 2
            },
            Rotation {
                angle: control.angle
                origin.x: handleItem.width / 2
                origin.y: handleItem.height / 2
            }
        ]
    }
}*/
        
      //  }
        
        
}