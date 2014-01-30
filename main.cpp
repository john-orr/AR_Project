#include <iostream>

#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <GL/glew.h>
#include <GL/freeglut.h>

using namespace std;
using namespace cv;

VideoCapture cap;
Mat frame;
Mat perspective_warped_image;
Point2f  src_pts[4], dst_pts[4];

void overlayImage();
int ChessBoard(Mat image);

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace std;


static const char* pVS = "                                                    \n\
#version 330                                                                  \n\
                                                                              \n\
in vec3 vPosition;															  \n\
in vec4 vColor;																  \n\
out vec4 color;																 \n\
                                                                              \n\
                                                                               \n\
void main()                                                                     \n\
{                                                                                \n\
    gl_Position = vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);  \n\
	color = vColor;							\n\
}";

static const char* pFS = "                                              \n\
#version 330                                                            \n\
                                                                        \n\
out vec4 FragColor;                                                      \n\
                                                                          \n\
void main()                                                               \n\
{                                                                          \n\
FragColor = vec4(1.0, 0.0, 0.0, 1.0);									 \n\
}";


// Shader Functions- click on + to expand
#pragma region SHADER_FUNCTIONS
static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
{
	// create a shader object
    GLuint ShaderObj = glCreateShader(ShaderType);

    if (ShaderObj == 0) {
        fprintf(stderr, "Error creating shader type %d\n", ShaderType);
        exit(0);
    }
	// Bind the source code to the shader, this happens before compilation
	glShaderSource(ShaderObj, 1, (const GLchar**)&pShaderText, NULL);
	// compile the shader and check for errors
    glCompileShader(ShaderObj);
    GLint success;
	// check for shader related errors using glGetShaderiv
    glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar InfoLog[1024];
        glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
        fprintf(stderr, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
        exit(1);
    }
	// Attach the compiled shader object to the program object
    glAttachShader(ShaderProgram, ShaderObj);
}

GLuint CompileShaders()
{
	//Start the process of setting up our shaders by creating a program ID
	//Note: we will link all the shaders together into this ID
    GLuint shaderProgramID = glCreateProgram();
    if (shaderProgramID == 0) {
        fprintf(stderr, "Error creating shader program\n");
        exit(1);
    }

	// Create two shader objects, one for the vertex, and one for the fragment shader
    AddShader(shaderProgramID, pVS, GL_VERTEX_SHADER);
    AddShader(shaderProgramID, pFS, GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };


	// After compiling all shader objects and attaching them to the program, we can finally link it
    glLinkProgram(shaderProgramID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(shaderProgramID, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(stderr, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

	// program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
    glValidateProgram(shaderProgramID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(shaderProgramID, GL_VALIDATE_STATUS, &Success);
    if (!Success) {
        glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(stderr, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }
	// Finally, use the linked shader program
	// Note: this program will stay in effect for all draw calls until you replace it with another or explicitly disable its use
    glUseProgram(shaderProgramID);
	return shaderProgramID;
}
#pragma endregion SHADER_FUNCTIONS

// VBO Functions - click on + to expand
#pragma region VBO_FUNCTIONS
GLuint generateObjectBuffer(GLfloat vertices[], GLfloat colors[]) {
	GLuint numVertices = 3;
	// Genderate 1 generic buffer object, called VBO
	GLuint VBO;
 	glGenBuffers(1, &VBO);
	// In OpenGL, we bind (make active) the handle to a target name and then execute commands on that target
	// Buffer will contain an array of vertices 
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// After binding, we now fill our object with data, everything in "Vertices" goes to the GPU
	glBufferData(GL_ARRAY_BUFFER, numVertices*7*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	// if you have more data besides vertices (e.g., vertex colours or normals), use glBufferSubData to tell the buffer when the vertices array ends and when the colors start
	glBufferSubData (GL_ARRAY_BUFFER, 0, numVertices*3*sizeof(GLfloat), vertices);
	glBufferSubData (GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), numVertices*4*sizeof(GLfloat), colors);
return VBO;
}

void linkCurrentBuffertoShader(GLuint shaderProgramID){
	GLuint numVertices = 3;
	// find the location of the variables that we will be using in the shader program
	GLuint positionID = glGetAttribLocation(shaderProgramID, "vPosition");
	GLuint colorID = glGetAttribLocation(shaderProgramID, "vColor");
	// Have to enable this
	glEnableVertexAttribArray(positionID);
	// Tell it where to find the position data in the currently active buffer (at index positionID)
    glVertexAttribPointer(positionID, 3, GL_FLOAT, GL_FALSE, 0, 0);
	// Similarly, for the color data.
	glEnableVertexAttribArray(colorID);
	glVertexAttribPointer(colorID, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(numVertices*3*sizeof(GLfloat)));
}
#pragma endregion VBO_FUNCTIONS


void display(){

	cap >> frame;

	perspective_warped_image = Mat::zeros(frame.rows, frame.cols, CV_8UC3);

	glClear(GL_COLOR_BUFFER_BIT);
	// NB: Make the call to draw the geometry in the currently activated vertex buffer. This is where the GPU starts to work!
	glDrawArrays(GL_TRIANGLES, 0, 3);
    glutSwapBuffers();

	unsigned char* buffer = new unsigned char[frame.cols*frame.rows*3];
	glReadPixels(0, 0, frame.cols, frame.rows, GL_BGR, GL_UNSIGNED_BYTE, buffer);
	Mat image(frame.rows, frame.cols, CV_8UC3, buffer);

	src_pts[0] =  Point2f(0, 0);
	src_pts[1] =  Point2f(image.cols, 0);
	src_pts[2] =  Point2f(0, image.rows);
	src_pts[3] =  Point2f(image.cols, image.rows);

	if(ChessBoard(image))
	{
		overlayImage();
	}

	imshow("Show Image", frame);

	glutPostRedisplay();
}


void init()
{	
	// Create 3 vertices that make up a triangle that fits on the viewport 
	GLfloat vertices[] = {-1.0f, -1.0f, 0.0f,
			1.0f, -1.0f, 0.0f,
			0.0f, 1.0f, 0.0f};
	// Create a color array that identfies the colors of each vertex (format R, G, B, A)
	GLfloat colors[] = {0.0f, 1.0f, 0.0f, 1.0f,
			1.0f, 0.0f, 0.0f, 1.0f,
			0.0f, 0.0f, 1.0f, 1.0f};
	// Set up the shaders
	GLuint shaderProgramID = CompileShaders();
	// Put the vertices and colors into a vertex buffer object
	generateObjectBuffer(vertices, colors);
	// Link the current buffer to the shader
	linkCurrentBuffertoShader(shaderProgramID);	
}

int main(int argc, char** argv)
{
	cap = VideoCapture(0);
	cap >> frame;

	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
    glutInitWindowSize(frame.cols, frame.rows);
    glutCreateWindow("Hello Triangle");
	// Tell glut where the display function is
	glutDisplayFunc(display);

	 // A call to glewInit() must be done after glut is initialized!
    GLenum res = glewInit();
	// Check for any errors
    if (res != GLEW_OK) {
      fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
      return 1;
    }
	// Set up your objects and shaders
	init();
	// Begin infinite event loop
	glutMainLoop();
    return 0;

}

int ChessBoard(Mat image)
{
	Size patternsize(7,7); //interior number of corners
	Mat gray;
	vector<Point2f> corners; //this will be filled by the detected corners
	bool patternfound;

	cvtColor(frame, gray, CV_BGR2GRAY); //source image
		
	//CALIB_CB_FAST_CHECK saves a lot of time on images
	//that do not contain any chessboard corners
	patternfound = findChessboardCorners(gray, patternsize, corners,/*CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ */CALIB_CB_FAST_CHECK);

	if(patternfound)
	{
		//cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		dst_pts[0] =  corners[0];
		dst_pts[1] =  corners[6];
		dst_pts[2] =  corners[42];
		dst_pts[3] =  corners[48];

		Mat perspective_matrix = getPerspectiveTransform(src_pts, dst_pts);
		warpPerspective(image, perspective_warped_image, perspective_matrix, perspective_warped_image.size());

		return 1;
	}
	return 0;
}

void overlayImage()
{
	vector<Mat> bgr;

	split(perspective_warped_image, bgr);
	//cvtColor(perspective_warped_image, tmp_gray, CV_BGR2GRAY);

	for(int i=0;i<frame.cols;i++)
	{
		for(int j=0;j<frame.rows;j++)
		{
			uchar blue = bgr[0].at<uchar>(j,i);
			uchar green = bgr[1].at<uchar>(j,i);
			uchar red = bgr[2].at<uchar>(j,i);

			if(!(blue == 0 && green == 0 && red == 0))
			{
				frame.at<Vec3b>(j,i)[0] = perspective_warped_image.at<Vec3b>(j,i)[0];
				frame.at<Vec3b>(j,i)[1] = perspective_warped_image.at<Vec3b>(j,i)[1];
				frame.at<Vec3b>(j,i)[2] = perspective_warped_image.at<Vec3b>(j,i)[2];
				perspective_warped_image.at<Vec3b>(j,i) = 255;
			}
		}
	}
} 