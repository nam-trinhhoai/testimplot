/****************************************************************************
**
** Copyright (C) 2015 Klaralvdalens Datakonsult AB (KDAB).
** Contact: https://www.qt.io/licensing/
**
** This file is part of the Qt3D module of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** Commercial License Usage
** Licensees holding valid commercial Qt licenses may use this file in
** accordance with the commercial license agreement provided with the
** Software or, alternatively, in accordance with the terms contained in
** a written agreement between you and The Qt Company. For licensing terms
** and conditions see https://www.qt.io/terms-conditions. For further
** information use the contact form at https://www.qt.io/contact-us.
**
** BSD License Usage
** Alternatively, you may use this file under the terms of the BSD license
** as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of The Qt Company Ltd nor the names of its
**     contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/


import QtQuick.Scene3D 2.0
import Qt3D.Render 2.15

import Qt3D.Core 2.15
import Qt3D.Input 2.0
import Qt3D.Extras 2.15

import QtQuick 2.0


Item {
    Text {
        text: "Click me!"
        anchors.top: parent.top
        anchors.topMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter

        MouseArea {
            anchors.fill: parent
            onClicked: animation.start()
        }
    }

    Text {
        text: "Multisample: " + scene3d.multisample
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 10
        anchors.horizontalCenter: parent.horizontalCenter

        MouseArea {
            anchors.fill: parent
            onClicked: scene3d.multisample = !scene3d.multisample
        }
    }

    Rectangle {
        id: scene
        anchors.fill: parent
        anchors.margins: 50
        color: "darkRed"

       /* transform: Rotation {
            id: sceneRotation
            axis.x: 1
            axis.y: 0
            axis.z: 0
            origin.x: scene.width / 2
            origin.y: scene.height / 2
        }*/

        Scene3D {
            id: scene3d
            anchors.fill: parent
            anchors.margins: 10
            focus: true
            aspects: ["input", "logic"]
            cameraAspectRatioMode: Scene3D.AutomaticAspectRatio

           Entity {
    id: rootEntity
    property RenderCapabilities capabilities : renderSettings.renderCapabilities

   // property bool validBounds: sphereMesh.implicitPointsValid
   // property vector3d sphereMinPt: sphereMesh.implicitMinPoint
   // property vector3d sphereMaxPt: sphereMesh.implicitMaxPoint

    Camera {
        id: camera
        projectionType: CameraLens.PerspectiveProjection
        fieldOfView: 45
        nearPlane : 0.1
        farPlane : 1000.0
        position: Qt.vector3d( 0.0, 0.0, 40.0 )
        upVector: Qt.vector3d( 0.0, 1.0, 0.0 )
        viewCenter: Qt.vector3d( 0.0, 0.0, 0.0 )
    }

    FirstPersonCameraController { camera: camera }

    components: [
        RenderSettings {
            id: renderSettings
            activeFrameGraph: ForwardRenderer {
                camera: camera
                clearColor: "transparent"
                showDebugOverlay: true
            }
        },
        InputSettings { }
    ]

   /* PhongMaterial {
        id: material
    }

    TorusMesh {
        id: torusMesh
        radius: 5
        minorRadius: 1
        rings: 100
        slices: 20
    }

    Transform {
        id: torusTransform
        scale3D: Qt.vector3d(1.5, 1, 0.5)
        rotation: fromAxisAndAngle(Qt.vector3d(1, 0, 0), 45)
    }

    Entity {
        id: torusEntity
        components: [ torusMesh, material, torusTransform ]
    }

    SphereMesh {
        id: sphereMesh
        radius: 3
        generateTangents: true
    }

    Transform {
        id: sphereTransform
        property real userAngle: 0.0
        matrix: {
            var m = Qt.matrix4x4();
            m.rotate(userAngle, Qt.vector3d(0, 1, 0))
            m.translate(Qt.vector3d(20, 0, 0));
            return m;
        }
    }

    NumberAnimation {
        target: sphereTransform
        property: "userAngle"
        duration: 10000
        from: 0
        to: 360

        loops: Animation.Infinite
        running: true
    }

    Entity {
        id: sphereEntity
        components: [ sphereMesh, material, sphereTransform ]
    }*/
}

        }
    }

    Rectangle {
        radius: 10
        color: "#aaffffff"
        border.width: 1
        border.color: "black"
        width: childrenRect.width + anchors.margins
        height: childrenRect.height + anchors.margins
        anchors.bottom: parent.bottom
        anchors.right: parent.right
        anchors.margins: 20

        Column {
            x: parent.anchors.margins / 2
            y: x

            Text { text: "Vendor: " + rootEntity.vendor }
            Text { text: "Renderer: " + rootEntity.renderer }
            Text { text: "Driver Version: " + rootEntity.driverVersion }
            Text { text: "GL Version: " + rootEntity.majorVersion + "." + rootEntity.minorVersion }
           // Text { text: "Profile: " + (rootEntity.profile === Render.CoreProfile ? "Core" : (rootEntity.profile === Render.CompatibilityProfile ? "Compatibility" : "Unknown")) }
            Text { text: "GLSL Version: " + rootEntity.glslVersion }
            //Text { text: "Extensions: " + (rootEntity.extensions.length ? "" : "None") }
          /*  ListView {
                model: rootEntity.extensions
                delegate: Text { text: "  " + model.modelData }
                width: parent.width
                height: 100
                visible: rootEntity.extensions.length > 0
                clip: true
            }*/
            Text { text: "Max Texture Size: " + rootEntity.capabilities.maxTextureSize + "\nMax Texture Units: " + rootEntity.capabilities.maxTextureUnits + "\nMax Texture Layers: " + rootEntity.capabilities.maxTextureLayers }
            Text { text: "Supports UBO: " + rootEntity.capabilities.supportsUBO }
            Text { text: "  Max UBO Size: " + rootEntity.capabilities.maxUBOSize + "\n  Max UBO Bindings: " + rootEntity.capabilities.maxUBOBindings; visible: rootEntity.capabilities.supportsUBO }
            Text { text: "Supports SSBO: " + rootEntity.capabilities.supportsSSBO }
            Text { text: "  Max SSBO Size: " + rootEntity.capabilities.maxSSBOSize + "\n  Max SSBO Bindings: " + rootEntity.capabilities.maxSSBOBindings; visible: rootEntity.capabilities.supportsSSBO }
            Text { text: "Supports Image Store: " + rootEntity.capabilities.supportsImageStore }
            Text { text: "  Max Image Units: " + rootEntity.capabilities.maxImageUnits; visible: rootEntity.capabilities.supportsImageStore }
            Text { text: "Supports Compute Shaders: " + rootEntity.capabilities.supportsCompute }
            Text { text: "  Max Work Group Size: " + rootEntity.capabilities.maxWorkGroupSizeX + ", " + rootEntity.capabilities.maxWorkGroupSizeY + ", " + rootEntity.capabilities.maxWorkGroupSizeZ; visible: rootEntity.capabilities.supportsCompute }
            Text { text: "  Max Work Group Count: " + rootEntity.capabilities.maxWorkGroupCountX + ", " + rootEntity.capabilities.maxWorkGroupCountY + ", " + rootEntity.capabilities.maxWorkGroupCountZ; visible: rootEntity.capabilities.supportsCompute }
            Text { text: "  Max Invocations: " + rootEntity.capabilities.maxComputeInvocations; visible: rootEntity.capabilities.supportsCompute }
            Text { text: "  Max Shared Memory: " + rootEntity.capabilities.maxComputeSharedMemorySize; visible: rootEntity.capabilities.supportsCompute }
        }
    }

    Rectangle {
        radius: 10
        color: "#aaffffff"
        border.width: 1
        border.color: "black"
        width: childrenRect.width + anchors.margins
        height: childrenRect.height + anchors.margins
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.margins: 20
        visible:false // rootEntity.validBounds

        Column {
            x: parent.anchors.margins / 2
            y: x

          //  Text { text: "Sphere:\n  Min Extent: " + rootEntity.sphereMinPt }
          //  Text { text: "  Max Extent: " + rootEntity.sphereMaxPt }
        }
    }

   /* SequentialAnimation {
        id: animation

        RotationAnimation {
            to: 45
            duration: 1000
            target: sceneRotation
            property: "angle"
            easing.type: Easing.InOutQuad
        }
        PauseAnimation { duration: 500 }
        NumberAnimation {
            to: 0.5
            duration: 1000
            target: scene
            property: "scale"
            easing.type: Easing.OutElastic
        }
        PauseAnimation { duration: 500 }
        NumberAnimation {
            to: 1.0
            duration: 1000
            target: scene
            property: "scale"
            easing.type: Easing.OutElastic
        }
        PauseAnimation { duration: 500 }
        RotationAnimation {
            to: 0
            duration: 1000
            target: sceneRotation
            property: "angle"
            easing.type: Easing.InOutQuad
        }
    }*/
}
