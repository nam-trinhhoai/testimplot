#version 150 core

in vec3 vertexPosition;

uniform mat4 modelMatrix;
uniform mat4 modelView;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main() {

    // Remove translation from Camera
    mat4 modelV = mat4(mat3(viewMatrix)) * modelMatrix;

    // Remove rotation
    modelV[0][0] = modelMatrix[0][0];
    modelV[0][1] = modelMatrix[0][1];
    modelV[0][2] = modelMatrix[0][2];

    modelV[1][0] = modelMatrix[1][0];
    modelV[1][1] = modelMatrix[1][1];
    modelV[1][2] = modelMatrix[1][2];

    modelV[2][0] = modelMatrix[2][0];
    modelV[2][1] = modelMatrix[2][1];
    modelV[2][2] = modelMatrix[2][2];

    gl_Position = vec4(modelV * vec4(vertexPosition, 1.0));
}
