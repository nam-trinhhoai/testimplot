#version 330 core

in vec2 vertexPosition;
uniform mat4 viewProjectionMatrix;
uniform mat4 transfoMatrix;

void main()
{
    gl_Position = viewProjectionMatrix *transfoMatrix*vec4(vertexPosition.x,vertexPosition.y, 0.0, 1.0);
}
