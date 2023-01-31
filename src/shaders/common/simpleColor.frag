#version 330 core

out vec4 f_fragColor; // shader output color
uniform vec4 color; // shader output color

void main()
{  
    f_fragColor = vec4(color.r,color.g,color.b,color.a);
}
