#version 330 core

in vec2 vertexPosition;
uniform mat4 viewProjectionMatrix;
uniform mat4 transfoMatrix;

uniform float pointsize=2.0f;
void main()
{
    gl_Position = viewProjectionMatrix *transfoMatrix*vec4(vertexPosition, 0.0, 1.0);
    gl_PointSize =pointsize;
}
