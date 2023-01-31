#version 150 core

in vec3 vertexPosition;
in vec3 vertexColor;
out vec3 color;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main() {
    color = vertexColor;
    // Converting the viewMatrix to a mat3, then back to a mat4
    // removes the translation component from it
    gl_Position = vec4(mat4(mat3(viewMatrix)) * modelMatrix * vec4(vertexPosition, 1.0));
}
