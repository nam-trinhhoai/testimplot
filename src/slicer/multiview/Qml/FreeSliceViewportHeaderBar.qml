import QtQuick 2.10
import QtQuick.Layouts 1.3
import QtQuick.Controls
import QtQuick.Controls.Material 2.1
import Qt3D.Core 2.10 as Qt3D

RowLayout {
    id: root
    property int index: 0
    property vector3d viewAxis
    signal flip()
    signal recenter()

  /*  ComboBox {
        id: statesChooser
        width: 400
        Layout.alignment: Qt.AlignVCenter
        Layout.fillWidth: true
        Layout.minimumWidth: 300
        textRole: "Text"
        //model: _sceneManager.freeSliceStateStack

        // The current state is always at the last position
        currentIndex: count - 1
        delegate: ItemDelegate {
            contentItem: Text {
                text: model.Text
                color: "black"
                font: statesChooser.font
                verticalAlignment: Text.AlignVCenter
            }
        }
        onActivated: _sceeManager.restoreFreeSliceState(index)
    }*/
    ToolButton {
        id: flipButton
        Layout.alignment: Qt.AlignVCenter
        icon.source: "qrc:/slicer/icons/fontawesome/flip.svg"
        onClicked: root.flip()
    }
    ToolButton {
        id: recenterSeedPointButton
        Layout.alignment: Qt.AlignVCenter
        icon.source: "qrc:/slicer/icons/fontawesome/crosshairs-solid.svg"
        onClicked: root.recenter()
    }
    ToolButton {
        id: minusButton
        Layout.alignment: Qt.AlignVCenter
        text: "<"
        onClicked: freeSliceSlider.__minus__()
    }

  /*  FreeSliceSlider {
        id: freeSliceSlider
        Layout.alignment: Qt.AlignVCenter
      //  property Qt3D.Transform sliceTransform: {return  _sceneManager.freeSliceInfo.mainSlice.transform}
        //property vector3d  slicesOffsets: {return  _sceneManager.freeSliceInfo.sliderOffsets}
        readonly property vector3d sliceAxis: Qt.vector3d(1,0,0) //{return  _sceneManager.freeSliceInfo.firstOrthogonalSlice.axis}
        readonly property vector3d sliceAxis2: Qt.vector3d(0,1,0) //{return  _sceneManager.freeSliceInfo.mainSlice.axis}
        readonly property vector3d sliceAxis3:Qt.vector3d(0,0,1) // {return  _sceneManager.freeSliceInfo.secondOrthogonalSlice.axis}

        slider_index:index

        freeSliceAxis: viewAxis
        originalAxis:
        {
            if (index === 0) { return Qt.vector3d(1.0, 0.0, 0.0)}
            if (index === 1) { return Qt.vector3d(0.0, 1.0, 0.0)}
            return Qt.vector3d(0.0, 0.0, 1.0)
        }

        //Connections {
        //    target: _sceneManager
        //    onPaintPointPositionChanged:
         //   {
               // console.log("Paint position changed: " + _sceneManager.paintPointPosition)
         //       var dims = _sceneManager.dimensions
         //       var paintPointPos = _sceneManager.paintPointPosition
         //       var pos = Qt.vector3d(Math.max(-dims.x/2.0,Math.min(dims.x/2.0, paintPointPos.x - dims.x/2.0)),
          //                            Math.max(-dims.y/2.0,Math.min(dims.y/2.0, paintPointPos.y - dims.y/2.0)),
         //                             Math.max(-dims.z/2.0,Math.min(dims.z/2.0, paintPointPos.z - dims.z/2.0)))

         //       _sceneManager.freeSliceInfo.mainSlice.transform.translation = pos
         //   }
     //   }

     //  }

        freeSliceCenter: {

			return Qt.vector3d(0,0,0)
            if (index === 0)
            {
                return _sceneManager.freeSliceInfo.mainSlice.center
            }
            if (index === 1)
            {
                return _sceneManager.freeSliceInfo.firstOrthogonalSlice.center
            }
            return _sceneManager.freeSliceInfo.secondOrthogonalSlice.center
        }


        onCenterMoved:
        {
            var offsets = _sceneManager.freeSliceInfo.sliderOffsets
            if (index === 0)
            {
                volumetricCube.freeSlice.mainPickingEntity.picker.select()
                if(offsets.x !== value)
                    offsets.x = value
            }
            else if (index === 1)
            {
                volumetricCube.freeSlice.firstOrthogonalPickingEntity.picker.select()
                if(offsets.y !== value)
                    offsets.y = value

            }
            else
            {
                volumetricCube.freeSlice.secondOrthogonalPickingEntity.picker.select()
                if(offsets.z !== value)
                    offsets.z = value
            }


            var c1 = sliceAxis1.times(offsets.x)
            var c2 = sliceAxis2.times(offsets.y)
            var c3 = sliceAxis3.times(offsets.z)

            var new_transl = c1.plus(c2).plus(c3)


            _sceneManager.freeSliceInfo.sliderOffsets = offsets
            sliceTransform.translation = new_transl


        }

    }
*/
    ToolButton {
        id: plusButton
        Layout.alignment: Qt.AlignVCenter
        text: ">"
        onClicked: freeSliceSlider.__plus__()
    }
    
     ToolButton {
        id: playButton
        Layout.alignment: Qt.AlignVCenter
        text: ">>"
        onClicked:
        {
        
        	console.log(" clicked!");
        	 timer.start();
       	}
    }
    
    Timer {
    	id:timer
        interval: 30; running: false;repeat: true
        onTriggered: 
        {
        	//console.log("timer ok");
        	
        	
        	if(freeSliceSlider.value<1.0)	freeSliceSlider.value  = freeSliceSlider.value+0.005;
        	else freeSliceSlider.value =0.0;
        freeSliceSlider.play();



        }
    }
    

}
