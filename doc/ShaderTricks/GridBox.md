Shader Generated Grid Cube
====================================

In this document I want do describe a little shader snippet which generates grid
cube on the fly using OpenGL and GLSL.

// TODO: Insert sample image here

Basic Desgin
------------

Traditionally, you would use line geometry stored in buffers. However, applying
the possibilities of modern OpenGL we can create the same cube on the fly.
For this we start by setting up OpenGL to allows us to render without buffers.
In order to do so, we need to generate an empty vertex array object (VAO).

```cpp
    GLuint vao;
    glGenVertexArray(1, &vao);
```

This allows us to execute a generating shader:

```cpp
    // Bind the empty VAO
    glBindVertexArray(vao);

    // Bind the shader program
    glBindProgram(prg);

    // Execute the shader
    glDrawArrays(GL_POINTS, 0, N);
```
Next, let's think about how to design a shader that generates the cube and
how the interface should look like. Obviously, the information we can use are
the two built-in variables:

```GLSL
    gl_VertexID // Index of each processed vertex per instance
    gl_InvocationID // Index of the processed instance
````

The basic idea is to define a loop for each dimension, and then sweeping these
loops along each dimension. 

// TODO: Insert concept drawing

The individual loops are here-by modelled using four points by using
`GL_LINES_ADJACENCY`:
```GLSL
	// Primitive 0: Along x-direction
	// Primitive 1: Along y-direction
	// Primitive 2: Along z-direction
	int primitiveID = gl_VertexID / 4;
    
	// Vertex index in the loop
	int nodeID = gl_VertexID % 4;
```

We will use the `nodeID` and `primitiveID` to generate the loops and thereafter
`gl_InvocationID` to sweep them along a dimension. Finally, the computed
position is transformed to clip space:

```GLSL

	// Select correct axis', 'axisOne' denotes the direction the loop is moving
	// along
	int axisOne =  primitiveID;
	int axisTwo = (primitiveID + 1) % 3;
	int axisTre = (primitiveID + 2) % 3;

	// Accumulate the position of the node,
    // where 'Axis' is an array of 3 vec3 containing the axis' of the cube in
    // model space and 'Size' is the length of a cube side. 
	vec3 pos = Origin;
	switch (nodeID)
	{
	case 0:
		break;
	case 1:
		pos += Size * Axis[axisTwo];
		break;
	case 2:
		pos += Size * Axis[axisTwo];
		pos += Size * Axis[axisTre];
		break;
	case 3:
		pos += Size * Axis[axisTre];
		break;
	}

	// Displace the grid according to the axis set by primitiveID.
    // Here, 'StepSize' denotes the distance between two grid lines.
	pos += float(gl_InstanceID) * StepSize * Axis[axisOne];

	// Transform the point to view space
	gl_Position = ModelViewProjectionMatrix * vec4(pos, 1);
``` 

Next, we need to convert each of the generated primitives (remember we are using
`GL_LINES_ADJACENCY`) into actual line primitives. In order to accomplish this
we will employ a simple geometry shader:

```GLSL
    layout(lines_adjacency) in;
    layout(invocations = 4) in;
    layout(line_strip, max_vertices = 2) out;

    void main()
    {
        gl_Position = gl_in[gl_InvocationID].gl_Position;
        EmitVertex();

        gl_Position = gl_in[(gl_InvocationID + 1) % 4].gl_Position;
        EmitVertex();

        EndPrimitive();
    }
```

In this shader I am using the geometry shader instancing feature, which allows
me to keep the actual shader code simpler than without it. For each primitive
the shader will be invoked once (four tiems in total `invocations = 4`) and
generate a single line primitive with two vertices. Besides, this is also 
supposed to be more efficient than emiting all the primitives at once.

The final invocation on the host side know looks as follows, where `N` is the
number of cells per cube dimension: 

```cpp
    // Execute the shader
	// 3 line-loops with 4 points, N replications of the loops per dimension
    glDrawArraysInstanced(GL_LINES_ADJACENCY, 0, 12, N + 1);
```

Using Separate Colours for each Dimension
-----------------------------------------

So far we the rendered grid cube has only a single color. In this section, I
will extend the shader a bit to allow us to select a color for each dimension.
Instead of just emitting the position of each vertex, we will also color value:

```GLSL
    // Here 'Out' is an output block with a variable 'Colour' and
    // 'Colours' is a list of three colour values (one per dimension)
    switch (nodeID)
	{
	case 0:
		Out.Colour = Colours[axisTre];
		break;
	case 1:
		pos += size * Axis[axisTwo];
		Out.Colour = Colours[axisTwo];
		break;
	case 2:
		pos += size * Axis[axisTwo];
		pos += size * Axis[axisTre];
		Out.Colour = Colours[axisTre];
		break;
	case 3:
		pos += size * Axis[axisTre];
		Out.Colour = Colours[axisTwo];
		break;
	}
```

In the geometry shader these colour values will then be reused such that the
first value of each input line primitive will define the colour of the entire
output primitive.

// TODO final image of grid cube with colours
