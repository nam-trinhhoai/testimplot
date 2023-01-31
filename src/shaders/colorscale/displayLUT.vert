#version 330 core

in vec2 vertexPosition;
in vec2 textureCoordinates;
out vec2 v_textureCoordinates;

void main()
{
    v_textureCoordinates = vec2(textureCoordinates);
    gl_Position = vec4(vertexPosition, 0.0, 1.0);
}
