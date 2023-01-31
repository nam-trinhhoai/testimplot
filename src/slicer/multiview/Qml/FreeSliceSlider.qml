import QtQuick 2.12
import QtQuick.Controls 2.4
import QtQuick.Controls.Material 2.1
import Murat 1.0

Slider {
    property vector3d freeSliceCenter
    property vector3d freeSliceAxis
    property int slider_index

    readonly property vector3d axisNormalized: freeSliceAxis.normalized()
    property vector3d originalAxis: freeSliceAxis.originalAxis
    readonly property vector3d halfExtent:root3D.dimensions// root3D.dimensions.times(0.5).times(_sceneManager.dataSetVoxelSize)
    property vector3d minPt
    property vector3d maxPt

    readonly property vector3d offsets: Qt.vector3d(0,0,0) //_sceneManager.freeSliceInfo.sliderOffsets

    //signal centerMoved(vector3d newCenter)
    signal centerMoved(real value)


    onFreeSliceCenterChanged: calculateMinMax()
    onAxisNormalizedChanged: calculateMinMax()
    onHalfExtentChanged:
    {
        calculateMinMax()
        maxLength = maxPt.minus(minPt).length()
        centerMoved(maxLength * value-maxLength/2)
    }

    onOffsetsChanged: calculateNewValue()

    property real maxLength: maxPt.minus(minPt).length()

    readonly property real currentLength: {
        var dot = maxPt.minus(minPt).dotProduct(freeSliceCenter.minus(minPt))
        if (dot < 0)
            return 0
        if (dot > maxLength * maxLength)
            return maxLength
        return freeSliceCenter.minus(minPt).length()
    }

    function calculateNewValue()
    {

        if(index == 0)
        {
            var vx = (offsets.x + maxLength/2)/maxLength
            value = (Math.max(Math.min(vx, 1.0), 0.0))
        }
        else if(index == 1)
        {
            var vy = (offsets.y + maxLength/2)/maxLength
            value = (Math.max(Math.min(vy, 1.0), 0.0))
        }
        else
        {
            var vz = (offsets.z + maxLength/2)/maxLength
            value = (Math.max(Math.min(vz, 1.0), 0.0))
        }

    }

    function calculateMinMax()
    {
        // calculate the intersection points between the volume bounding box and the ray passing
        // through the slice's center with the slice's axis
        var result = MathUtils.lineAABBIntersection(freeSliceCenter, axisNormalized, halfExtent.times(-1), halfExtent)
        minPt = result.minPoint;
        maxPt = result.maxPoint;
    }
    
    
    function __minus__()
    {
    	var newCenter = minPt.plus(axisNormalized.times(maxLength * value - 1))
        centerMoved(newCenter)
    }
    
    function __plus__()
    {
    	var newCenter = minPt.plus(axisNormalized.times(maxLength * value + 1))
        centerMoved(newCenter)
    }
    
    
      function play()
    {
   		var newCenter = maxLength * value-maxLength/2
    //	var newCenter = minPt.plus(axisNormalized.times(maxLength * value + 1))
        centerMoved(newCenter)
    }

    onMoved: {
        //var newCenter = minPt.plus(axisNormalized.times(maxLength * value))
        //var newCenter = minPt.plus(originalAxis.times(maxLength * value))
     
        //centerMoved(newCenter)
        centerMoved(maxLength * value-maxLength/2)

    }

    from: 0
    to: 1
    value: currentLength / maxLength
}
