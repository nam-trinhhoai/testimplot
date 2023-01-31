#version 150 core

in vec3 vertexPosition;
in vec2 vertexTexCoord;

out vec3 worldPosition;
out vec2 texCoord;




uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelView;
uniform mat4 projectionMatrix;
uniform float aspectRatio;
uniform float screenWidth;
//uniform float bottom;
//uniform float left;
//uniform float right;
//uniform float top;

uniform mat4 mvp;

void main()
{
    texCoord = vertexTexCoord;




    // Transform position, normal, and tangent to world coords
    //worldPosition = vec3(modelMatrix * vec4(vertexPosition, 1.0));
    
    worldPosition = vertexPosition - vec3(0,0,10000);
    
    
    //vec3 viewScale = vec3(1.0f/viewMatrix[0][0], 1.0f/viewMatrix[0][0], 1.0f/viewMatrix[0][0]);
    
    float scale = 20.0/screenWidth; //0.05;
    vec4 viewpos = viewMatrix * vec4(vertexPosition * scale, 0.0);
    //vec4 viewpos = vec4(vertexPosition * scale, 0.0);
    viewpos.x += scale*1.5; //0.04;
    viewpos.y += scale*1.5; //0.04;

    viewpos.z -= 100;
    viewpos.w = 1;
    
    
    const float right = 1.0;
    const float bottom = 0;
    const float left = 0;
    float top = right / aspectRatio;
    const float far = 20000.0;
    const float near = -1000.0;
    
    
    
    mat4 ortho = mat4(
    vec4(2.0 / (right - left), 0, 0, 0),
    vec4(0, 2.0 / (top - bottom), 0, 0),
    vec4(0, 0, -2.0 / (far - near), 0),
    vec4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(far + near) / (far - near), 1)
    );

    
    

    // Calculate vertex position in clip coordinates
    //gl_Position = projectionMatrix * viewpos;
    gl_Position = ortho * viewpos;
}
