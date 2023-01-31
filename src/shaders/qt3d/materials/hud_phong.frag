#version 150 core

in vec2 texCoord;

out vec4 fragColor;

uniform vec3 maincolor;
uniform sampler2D diffuseTexture;
uniform float aspectRatio;
uniform float bottom;
uniform float left;
uniform float right;
uniform float top;


void main()
{
    //output color from material
    
    vec4 diffuse = texture(diffuseTexture, texCoord);
       
    fragColor = vec4(maincolor, 1.0) * diffuse;
    //fragColor = vec4(left/2.0, 0 ,0,1);
    
    gl_FragDepth = gl_FragCoord.z;
}
