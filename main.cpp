#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <iostream>
#include <Windows.h>
#include <process.h>
#include "opencv2/core/opengl_interop.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "maths_funcs.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

using namespace std;
using namespace cv;

VideoCapture cap;
Mat frame;
Mat perspective_warped_image;
Point2f  src_pts[4], dst_pts[4];
GLuint shaderProgramID;
unsigned char* buffer;
GLfloat vertices[108];
GLuint numVertices;
GLuint bgShader;
GLuint VBOQ; // VBO for the Quad
GLuint VBO; // vbo for the cube
int endOfColours,endOfQuad;
int selected_cube;

vector<vec3> cubes;
//Calibration variables
bool calibrateZ = false;
double markerZvalue = -1;
double baseRadius = 0;
bool calibrated = false;
bool grabbed = false;
bool hasInitialized = false;

Size patternsize(7,7); //interior number of corners
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
vector<Mat> rvecs, tvecs;
vector<vector<Point2f>> imagePoints;
vector<vector<Point3f>> objectPoints;
Mat projection = Mat::zeros(4, 4, CV_64F);;
Mat modelview = Mat::zeros(4, 4, CV_64F);;
Mat openGLtoCV;
mat4 modelV_mat4, view, persp_proj;
vec3 worldPos, closestPoint, grabbed_vertex, start, endPos;
int testImages = 0;
double zNear = 0.1;
double zFar = 500;

// file for the shaders to write to
FILE *ErrorTxt;

void overlayImage();
void ChessBoard();
void calibrateCameraMatrix();
float* getTransform(Mat& mat);
void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview);
GLfloat* convertMatrixType(const cv::Mat& m);

vec3 getClosest(Point pointerLoc);
float getDist(vec3 point, vec3 otherPoint);

Point object_pt;
bool found=false;
Mat object_mat;
Rect obj;
int radius;
Mat mask;
Mat getNormalizedRGB(const Mat& rgb);
bool find_rough(Mat src, Point& object_center, Rect& object);
float find_euclidian(float r, float g, float b, float r_t, float g_t, float b_t);

void thread(void* );
void chessboard_thread(void* );
GLfloat* modelV;
GLfloat* projV;

GLuint bg_tex; // texture that the open cv frame gets put into 

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace std;


#pragma region SHADER_FUNCTIONS
char* readShaderSource(const char* shaderFile) {   
    FILE* fp = fopen(shaderFile, "rb"); //!->Why does binary flag "RB" work and not "R"... wierd msvc thing?

    if ( fp == NULL ) { return NULL; }

    fseek(fp, 0L, SEEK_END);
    long size = ftell(fp);

    fseek(fp, 0L, SEEK_SET);
    char* buf = new char[size + 1];
    fread(buf, 1, size, fp);
    buf[size] = '\0';

    fclose(fp);

    return buf;
}


static void AddShader(GLuint ShaderProgram, const char* pShaderText, GLenum ShaderType)
{
	// create a shader object
    GLuint ShaderObj = glCreateShader(ShaderType);

    if (ShaderObj == 0) {
        fprintf(ErrorTxt, "Error creating shader type %d\n", ShaderType);
        exit(0);
    }
	const char* pShaderSource = readShaderSource( pShaderText);

	// Bind the source code to the shader, this happens before compilation
	glShaderSource(ShaderObj, 1, (const GLchar**)&pShaderSource, NULL);
	// compile the shader and check for errors
    glCompileShader(ShaderObj);
    GLint success;
	// check for shader related errors using glGetShaderiv
    glGetShaderiv(ShaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar InfoLog[1024];
        glGetShaderInfoLog(ShaderObj, 1024, NULL, InfoLog);
        fprintf(ErrorTxt, "Error compiling shader type %d: '%s'\n", ShaderType, InfoLog);
        exit(1);
    }
	// Attach the compiled shader object to the program object
    glAttachShader(ShaderProgram, ShaderObj);
}

GLuint CompileShaders()
{
	//Start the process of setting up our shaders by creating a program ID
	//Note: we will link all the shaders together into this ID
    shaderProgramID = glCreateProgram();
    if (shaderProgramID == 0) {
        fprintf(ErrorTxt, "Error creating shader program\n");
        exit(1);
    }

	// Create two shader objects, one for the vertex, and one for the fragment shader
    AddShader(shaderProgramID, "../Shaders/simpleVertexShader.txt", GL_VERTEX_SHADER);
    AddShader(shaderProgramID, "../Shaders/simpleFragmentShader.txt", GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };
	// After compiling all shader objects and attaching them to the program, we can finally link it
    glLinkProgram(shaderProgramID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(shaderProgramID, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(ErrorTxt, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

	// program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
    glValidateProgram(shaderProgramID);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(shaderProgramID, GL_VALIDATE_STATUS, &Success);
    if (!Success) {
        glGetProgramInfoLog(shaderProgramID, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(ErrorTxt, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }
	// Finally, use the linked shader program
	// Note: this program will stay in effect for all draw calls until you replace it with another or explicitly disable its use
    glUseProgram(shaderProgramID);
	return shaderProgramID;
}

GLuint compileQuadShaders()
{
	bgShader = glCreateProgram();
    if (bgShader == 0) {
        fprintf(ErrorTxt, "Error creating shader program\n");
        exit(1);
    }

	// Create two shader objects, one for the vertex, and one for the fragment shader
    AddShader(bgShader, "../Shaders/bgVertShader.txt", GL_VERTEX_SHADER);
    AddShader(bgShader, "../Shaders/bgFragShader.txt", GL_FRAGMENT_SHADER);

    GLint Success = 0;
    GLchar ErrorLog[1024] = { 0 };
	// After compiling all shader objects and attaching them to the program, we can finally link it
    glLinkProgram(bgShader);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(bgShader, GL_LINK_STATUS, &Success);
	if (Success == 0) {
		glGetProgramInfoLog(bgShader, sizeof(ErrorLog), NULL, ErrorLog);
		fprintf(ErrorTxt, "Error linking shader program: '%s'\n", ErrorLog);
        exit(1);
	}

	// program has been successfully linked but needs to be validated to check whether the program can execute given the current pipeline state
    glValidateProgram(bgShader);
	// check for program related errors using glGetProgramiv
    glGetProgramiv(bgShader, GL_VALIDATE_STATUS, &Success);
    if (!Success) {
        glGetProgramInfoLog(bgShader, sizeof(ErrorLog), NULL, ErrorLog);
        fprintf(ErrorTxt, "Invalid shader program: '%s'\n", ErrorLog);
        exit(1);
    }
	// Finally, use the linked shader program
	// Note: this program will stay in effect for all draw calls until you replace it with another or explicitly disable its use
    glUseProgram(bgShader);
	return bgShader;
}

#pragma endregion SHADER_FUNCTIONS

#pragma region VBO_FUNCTIONS
GLuint generateObjectBuffer(GLfloat vertices[], GLfloat colors[], GLfloat Quad[], GLfloat Quadtc[]) {
	numVertices = 36;
	int quadSize = 6;
	int nextStart;
	// Genderate 1 generic buffer object, called VBO
 	glGenBuffers(1, &VBO);
	// In OpenGL, we bind (make active) the handle to a target name and then execute commands on that target
	// Buffer will contain an array of vertices 
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// After binding, we now fill our object with data, everything in "Vertices" goes to the GPU
	glBufferData(GL_ARRAY_BUFFER, numVertices*7*sizeof(GLfloat), NULL, GL_STREAM_DRAW);
	// if you have more data besides vertices (e.g., vertex colours or normals), use glBufferSubData to tell the buffer when the vertices array ends and when the colors start
	glBufferSubData (GL_ARRAY_BUFFER, 0, numVertices*3*sizeof(GLfloat), vertices);
	nextStart = numVertices*3*sizeof(GLfloat);
	glBufferSubData (GL_ARRAY_BUFFER,nextStart, numVertices*4*sizeof(GLfloat), colors);
	nextStart += numVertices*4*sizeof(GLfloat);
	endOfColours = nextStart;
	glBufferSubData(GL_ARRAY_BUFFER, nextStart, quadSize*3*sizeof(GLfloat),Quad);
	nextStart += quadSize*3*sizeof(GLfloat);
	endOfQuad = nextStart; 
	glBufferSubData(GL_ARRAY_BUFFER,nextStart, quadSize*2*sizeof(GLfloat),Quadtc);
return VBO;
}

void linkCurrentBuffertoShader(GLuint shaderProgramID){
	GLuint numVertices = 36;
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
/*
	glUseProgram(bgShader);
	GLuint positionLoc = glGetAttribLocation(bgShader, "vPosition");
	GLuint tcLoc = glGetAttribLocation(bgShader, "tc");
	glEnableVertexAttribArray(positionLoc);
	glVertexAttribPointer(positionLoc,3,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(endOfColours));
	glEnableVertexAttribArray(tcLoc);
	glVertexAttribPointer(tcLoc,2,GL_FLOAT,GL_FALSE,0,BUFFER_OFFSET(endOfQuad));
	*/
}
GLuint generateQuadObjectBuffer(GLfloat vertices[], GLfloat tex[]) {
	numVertices = 6;
	// Genderate 1 generic buffer object, called VBO
	
 	glGenBuffers(1, &VBOQ);
	// In OpenGL, we bind (make active) the handle to a target name and then execute commands on that target
	// Buffer will contain an array of vertices 
	glBindBuffer(GL_ARRAY_BUFFER, VBOQ);
	// After binding, we now fill our object with data, everything in "Vertices" goes to the GPU
	glBufferData(GL_ARRAY_BUFFER, numVertices*7*sizeof(GLfloat), NULL, GL_STREAM_DRAW);
	glBufferSubData (GL_ARRAY_BUFFER, 0, numVertices*3*sizeof(GLfloat), vertices);
	glBufferSubData (GL_ARRAY_BUFFER, numVertices*3*sizeof(GLfloat), numVertices*2*sizeof(GLfloat), tex);
return VBOQ;
}

void linkQuadBuffertoShader(GLuint shaderProgramID){
	GLuint numVertices = 6;
	// find the location of the variables that we will be using in the shader program
	GLuint positionID = glGetAttribLocation(shaderProgramID, "vPosition");
	GLuint texID = glGetAttribLocation(shaderProgramID, "tc");
	// Have to enable this
	glEnableVertexAttribArray(positionID);
	// Tell it where to find the position data in the currently active buffer (at index positionID)
    glVertexAttribPointer(positionID, 3, GL_FLOAT, GL_FALSE, 0, 0);
	// Similarly, for the color data.
	glEnableVertexAttribArray(texID);
	glVertexAttribPointer(texID, 4, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(numVertices*3*sizeof(GLfloat)));
}
#pragma endregion VBO_FUNCTIONS

void translateVertex (vec3 vertex, vec3 vector){
	for (int i=0; i<36; i++){
		//edit the relevant vertices in the vertex array
		if (vertices[3*i] == vertex.v[0] && vertices[3*i+1] == vertex.v[1] && vertices[3*i+2] == vertex.v[2]){
					vertices[3*i] += vector.v[0];
					vertices[3*i+1] += vector.v[1];
					vertices[3*i+2] += vector.v[2];
		}
		//reload the vertex buffer
		glBufferSubData (GL_ARRAY_BUFFER, 0, numVertices*3*sizeof(GLfloat), vertices);
	}
}

vec3 convertToModelCoords(vec3 worldcoords){
	//make a mat4 version of the modelview matrix
	//this is now done in the chessboard function
	//then use that to create the model matrix of the untranslated cube
	mat4 model = modelV_mat4*scale(identity_mat4(), vec3(0.25, 0.25, 0.25));
	
	//derive inverse of the matrix
	mat4 model_inv = inverse(model);

	//undo the effects of those darn matrices
	vec4 vec4_worldcoords = vec4(worldcoords, 1);
	vec3 result = model_inv*vec4_worldcoords;

	return result;
}


int getClosestCube(){
	//find closest centre
	int closest = 0; 
	vec3 current;
	float currentDist;
	float closestDist = 500;
	for (int i=0; i<cubes.size(); i++)
	{
		currentDist = getDist(worldPos,cubes[i]);
		if(currentDist < closestDist)
		{
			closestDist = currentDist;
			closest = i; 
		}
	}

	return closest;
}

void drawALLTheCubes(mat4 original){
	int model_mat_location = glGetUniformLocation (shaderProgramID, "model");
	int selected_loc = glGetUniformLocation(shaderProgramID, "selected");
	selected_cube = getClosestCube();
	for (int i=0; i<cubes.size(); i++){
		mat4 model = translate(identity_mat4(), cubes[i]);
		model = original*scale(model, vec3(0.25, 0.25, 0.25));
		if(i == selected_cube)
		{
			glUniform1i(selected_loc, 1);
		}
		else 
		{
			glUniform1i(selected_loc, 0 );
		}
		glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, model.m);
		glDrawArrays(GL_TRIANGLES, 0, 36);
	}
}


void display(){

	cap >> frame;

	cap >> frame;
	/*
	if(found == false){
		//found = find_rough(frame, object_pt, obj);
	}
	*/
	if(found == true){
		circle(frame, object_pt, radius, CV_RGB(0,0,255), 1, 8, 0);
	}

	if(!calibrateZ && baseRadius == 0)
		{
			baseRadius = radius;
		}

	vec3 temp = getClosest(object_pt);
	//cout << "this is the closest point ";
	//print(temp);
	if (grabbed){
		endPos = worldPos;
		vec3 translation =  endPos - start;
		translateVertex(grabbed_vertex, translation);
		grabbed_vertex += translation;
		start = endPos;
	}
	perspective_warped_image = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
	flip(frame, frame, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// transfer contents of the cv:Mat to the texture

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,frame.cols,frame.rows,0,GL_BGR,GL_UNSIGNED_BYTE,frame.ptr());

	// draw the quad
	glUseProgram(bgShader);
	glBindTexture(GL_TEXTURE_2D,bg_tex);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glClear(GL_DEPTH_BUFFER_BIT);

	// rebind the cube vbo and shaders 

	glUseProgram(shaderProgramID);
	// draw the cube 
	int proj_mat_location = glGetUniformLocation (shaderProgramID, "proj");
	int model_mat_location = glGetUniformLocation (shaderProgramID, "model");

	
	if(!calibrated)
	{
		calibrateCameraMatrix();
		if(calibrated)
		_beginthread( chessboard_thread, 0, (void*)12 );
	} 

	if(hasInitialized)
	{
		glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, modelV);
		glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, projV);
	}

	int selected_loc = glGetUniformLocation(shaderProgramID, "selected");

	glUniform3f(selected_loc,temp.v[0],temp.v[1],temp.v[2]);
	drawALLTheCubes(modelV_mat4);
	glutSwapBuffers();


	glutPostRedisplay();
}

void init()
{	
#pragma region vertices 
	GLfloat temp[] = {  
							-1.0f,  1.0f, -1.0f, 
							1.0f,  1.0f, -1.0f, 
							1.0f, -1.0f, -1.0f, 
							1.0f, -1.0f, -1.0f,  
							-1.0f, -1.0f, -1.0f, 
							-1.0f,  1.0f, -1.0f, 

							-1.0f,  1.0f, 1.0f,  
							1.0f,  1.0f, 1.0f, 
							1.0f, -1.0f, 1.0f, 
							1.0f, -1.0f, 1.0f,  
							-1.0f, -1.0f, 1.0f, 
							-1.0f,  1.0f, 1.0f, 

							-1.0f,  1.0f, -1.0f,
							1.0f,  1.0f, -1.0f,
							1.0f,  1.0f, 1.0f, 
							1.0f,  1.0f, 1.0f,
							-1.0f,  1.0f, 1.0f, 
							-1.0f,  1.0f, -1.0f, 

							-1.0f,  -1.0f, -1.0f, 
							1.0f,  -1.0f, -1.0f, 
							1.0f,  -1.0f, 1.0f,
							1.0f,  -1.0f, 1.0f, 
							-1.0f,  -1.0f, 1.0f, 
							-1.0f,  -1.0f, -1.0f,

							-1.0f,   1.0f, 1.0f, 
							-1.0f,  -1.0f, 1.0f, 
							-1.0f,  -1.0f, -1.0f, 
							-1.0f,  -1.0f, -1.0f, 
							-1.0f,   1.0f, -1.0f, 
							-1.0f,   1.0f, 1.0f, 

							1.0f,   1.0f, 1.0f, 
							1.0f,  -1.0f, 1.0f, 
							1.0f,  -1.0f, -1.0f, 
							1.0f,  -1.0f, -1.0f, 
							1.0f,   1.0f, -1.0f, 
							1.0f,   1.0f, 1.0f
						};
	memcpy(vertices, temp, 108*sizeof(GLfloat)); 
	// Create a color array that identfies the colors of each vertex (format R, G, B, A)
	GLfloat colors[] = {1.0f, 0.0f, 1.0f, 1.0f,
						1.0f, 0.0f, 1.0f, 1.0f,
						1.0f, 0.0f, 1.0f, 1.0f,
						1.0f, 0.0f, 1.0f, 1.0f,
						1.0f, 0.0f, 1.0f, 1.0f,
						1.0f, 0.0f, 1.0f, 1.0f,

						1.0f, 0.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,

						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,

						0.0f, 1.0f, 0.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,

						1.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 1.0f, 0.0f, 1.0f,

						0.0f, 1.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 1.0f, 1.0f
	};

#pragma endregion vertices 

#pragma region quad
	
	GLfloat quad[] = 
	{
		-1.0f, -1.0f,0.0f,
		1.0f, -1.0f,0.0f,
		-1.0f,  1.0f,0.0f,

		-1.0f,  1.0f,0.0f,
		1.0f, -1.0f,0.0f,
		1.0f,  1.0f, 0.0f
	};

	GLfloat quad_tc[] = 
	{
		0.0f, 0.0f,
		1.0f, 0.0f,
		0.0f, 1.0f,

		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0
	};
	/*
	GLfloat quad[] = 
	{
		-1.0,1.0,0.0,
		1.0,1.0,0.0,
		1.0,-1.0,0.0,
		-1.0,-1.0,0.0
	};
	GLfloat quad_tc[] =
	{
		.0,1.0,
		1.0,1.0,
		1.0,0.0,
		0.0,0.0
	};
	*/

#pragma endregion quad 

	for (int x=0; x<4; x++){
		for (int y=0; y<4; y++){
			for (int z=0; z<4; z++){
				cubes.push_back(vec3(2*x, 2*y, 2*z));
			}
		}
	}
	// Set up the shaders
	shaderProgramID = CompileShaders();

	bgShader = compileQuadShaders();
	// Put the vertices and colors into a vertex buffer object
	generateObjectBuffer(vertices, colors,quad, quad_tc);
	// Link the current buffer to the shader
	linkCurrentBuffertoShader(shaderProgramID);	


	glGenTextures(1,&bg_tex);
	glBindTexture(GL_TEXTURE_2D, bg_tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,frame.cols,frame.rows,0,GL_BGR,GL_UNSIGNED_BYTE,frame.ptr());

	glUseProgram (shaderProgramID);

	int proj_mat_location = glGetUniformLocation (shaderProgramID, "proj");
	int view_mat_location = glGetUniformLocation (shaderProgramID, "view");
	int model_mat_location = glGetUniformLocation (shaderProgramID, "model");


	persp_proj = perspective(170.0, (float)frame.cols/(float)frame.rows, 0.1, 500.0);
	view = identity_mat4();
	modelV_mat4 = identity_mat4();

	glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, persp_proj.m);
	glUniformMatrix4fv (view_mat_location, 1, GL_FALSE, view.m);
	glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, modelV_mat4.m);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
}

void keypress(unsigned char key, int x, int y){
	
	/*if (key == 'g')
	{
		cout << "Grabbed is now true.\n";
		grabbed = true;
		grabbed_vertex = closestPoint;
		start = worldPos;
	}
	if (key == 's')
	{
		cout << "Grabbed is now false.\n";
		grabbed = false;
		endPos = worldPos;
		vec3 translation =  endPos - start;
		translateVertex(grabbed_vertex, translation);
		start = endPos;
	}*/
	if (key == 'd'){
		cout << "Deleting a cube.\n";
		cubes.erase(cubes.begin() + selected_cube);
	}
}

int main(int argc, char** argv)
{
	ErrorTxt = fopen("error.txt","w");
	cap = VideoCapture(0);
	cap >> frame;
	cap >> frame;

	mask = Mat(frame.size(), CV_8UC1);
	_beginthread( thread, 0, (void*)12 );

	// Set up the window
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
    glutInitWindowSize(frame.cols, frame.rows);
    glutCreateWindow("AR");
	// Tell glut where the display function is
	glutDisplayFunc(display);
	glutKeyboardFunc(keypress);

	 // A call to glewInit() must be done after glut is initialized!
    GLenum res = glewInit();
	// Check for any errors
    if (res != GLEW_OK) {
      fprintf(stderr, "Error: '%s'\n", glewGetErrorString(res));
      return 1;
    }
	// Set up your objects and shaders
	init();
	glViewport (0, 0, frame.cols, frame.rows);
	// Begin infinite event loop
	glutMainLoop();

    return 0;

}

void thread(void* arg){
	while(true){
	found = find_rough(frame, object_pt, obj);
	}
}

void chessboard_thread(void* arg){
	cout<<"started chessboard thread"<<endl;
	while(true)
	{
		ChessBoard();
	}
}



void calibrateCameraMatrix()
{
	Mat gray;
	vector<Point2f> pointBuf;
	vector<Point3f> objectCorners;

	int squareLength = 1;

	cvtColor(frame, gray, CV_BGR2GRAY);

	bool patternfound = findChessboardCorners(gray, patternsize, pointBuf, CALIB_CB_FAST_CHECK);

	if(patternfound)
	{
		testImages++;
		cornerSubPix(gray, pointBuf, Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		imagePoints.push_back(pointBuf);

		for(int i=0; i<7; i++){
            for(int j=0;j<7;j++){
                objectCorners.push_back(cv::Point3f(float(i)*squareLength,float(j)*squareLength,0.0f));
            }
        }

		objectPoints.push_back(objectCorners);

		cout << "pointBuf: " << objectCorners.size() << endl;
		cout << "objectCorners: " << pointBuf.size() << endl;
		cout << "Images: " << testImages << endl;
		cout << "ImagePoints: " << imagePoints.size() << endl;
		cout << "ObjectPoints: " << objectPoints.size() << endl;
		cout << "************\n";


		calibrateCamera(objectPoints, imagePoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs, 0);
		calibrated = true;
		cout << "Just calibrated!\n";
	}

}

void ChessBoard() 
{
	Mat gray;
	vector<Point2f> points;				//this will be filled by the detected corners
	vector<Point2f> corners;
	vector<Point3f> Coords3d;
	Mat rotation;						// The calculated rotation of the chess board.
	Mat translation;					// The calculated translation of the chess board.
	double squareLength = 1;
	bool patternfound;
	cout << "Error 1:\n";
	cvtColor(frame, gray, CV_BGR2GRAY); //source image
	cout << "Error 2:\n";
	patternfound = findChessboardCorners(gray, patternsize, points, CALIB_CB_FAST_CHECK);
	cout << "Error 3:\n";
	if(patternfound)
	{

		corners.push_back(points[0]);
		corners.push_back(points[6]);
		corners.push_back(points[42]);
		corners.push_back(points[48]);

		for(int i = 0; i<2; i++){
            for(int j=0;j<2;j++){
                Coords3d.push_back(cv::Point3f(float(i)*squareLength,float(j)*squareLength,0.0f));
            }
        }

		solvePnP(Coords3d, corners, cameraMatrix, distCoeffs, rotation, translation);
		Mat rotationMatrix;
		Rodrigues(rotation, rotationMatrix);
		generateProjectionModelview(cameraMatrix, rotationMatrix, translation, projection, modelview);

		modelV = convertMatrixType(modelview);
		projV = convertMatrixType(projection);
		hasInitialized = true;

		if(!calibrateZ)
		{
			markerZvalue = modelview.at<double>(2,3);
			baseRadius = radius;
			calibrateZ = true;
		}

		modelV_mat4 = mat4(modelV[0], modelV[4], modelV[8], modelV[12], 
								modelV[1], modelV[5], modelV[9], modelV[13], 
								modelV[2], modelV[6], modelV[10], modelV[14], 
								modelV[3], modelV[7], modelV[11], modelV[15]);

	}
	cout << "Error 4:\n";
}

void overlayImage()
{
	vector<Mat> bgr;

	split(openGLtoCV, bgr);

	for(int i=0;i<frame.cols;i++)
	{
		for(int j=0;j<frame.rows;j++)
		{
			uchar blue = bgr[0].at<uchar>(j,i);
			uchar green = bgr[1].at<uchar>(j,i);
			uchar red = bgr[2].at<uchar>(j,i);

			if(!(blue == 0 && green == 0 && red == 0))
			{
				frame.at<Vec3b>(j,i)[0] = openGLtoCV.at<Vec3b>(j,i)[0];
				frame.at<Vec3b>(j,i)[1] = openGLtoCV.at<Vec3b>(j,i)[1];
				frame.at<Vec3b>(j,i)[2] = openGLtoCV.at<Vec3b>(j,i)[2];
				openGLtoCV.at<Vec3b>(j,i) = 255;
			}
		}
	}
} 

float* getTransform(Mat& mat)
{
	float transform_temp[] = {
								mat.at<double>(0,0), mat.at<double>(0,1), mat.at<double>(0,2), 0,
								mat.at<double>(1,0), mat.at<double>(1,1), mat.at<double>(1,2), 0,
								mat.at<double>(2,0), mat.at<double>(1,1), mat.at<double>(2,2), 0
							};
	return transform_temp;
}

void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview)
{
	typedef double precision;

 	projection.at<precision>(0,0) = 2*calibration.at<precision>(0,0)/640;//X
	projection.at<precision>(1,0) = 0;
	projection.at<precision>(2,0) = 0;
	projection.at<precision>(3,0) = 0;

	projection.at<precision>(0,1) = 0;
	projection.at<precision>(1,1) = 2*calibration.at<precision>(1,1)/480;//Y
	projection.at<precision>(2,1) = 0;
	projection.at<precision>(3,1) = 0;


	projection.at<precision>(0,2) = 1-2*calibration.at<precision>(0,2)/640;
	projection.at<precision>(1,2) = -1+(2*calibration.at<precision>(1,2)+2)/480;
	projection.at<precision>(2,2) = (zNear+zFar)/(zNear - zFar);
	projection.at<precision>(3,2) = -1;

	projection.at<precision>(0,3) = 0;
	projection.at<precision>(1,3) = 0;
	projection.at<precision>(2,3) = 2*zNear*zFar/(zNear - zFar);
	projection.at<precision>(3,3) = 0;


	modelview.at<precision>(0,0) = rotation.at<precision>(0,0);
	modelview.at<precision>(1,0) = rotation.at<precision>(1,0);
	modelview.at<precision>(2,0) = rotation.at<precision>(2,0);
	modelview.at<precision>(3,0) = 0;

	modelview.at<precision>(0,1) = rotation.at<precision>(0,1);
	modelview.at<precision>(1,1) = rotation.at<precision>(1,1);
	modelview.at<precision>(2,1) = rotation.at<precision>(2,1);
	modelview.at<precision>(3,1) = 0;

	modelview.at<precision>(0,2) = rotation.at<precision>(0,2);
	modelview.at<precision>(1,2) = rotation.at<precision>(1,2);
	modelview.at<precision>(2,2) = rotation.at<precision>(2,2);
	modelview.at<precision>(3,2) = 0;

	modelview.at<precision>(0,3) = translation.at<precision>(0,0);
	modelview.at<precision>(1,3) = translation.at<precision>(1,0);
	modelview.at<precision>(2,3) = translation.at<precision>(2,0);
	modelview.at<precision>(3,3) = 1;

	// This matrix corresponds to the change of coordinate systems.
	static double changeCoordArray[4][4] = {{1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}};
	static Mat changeCoord(4, 4, CV_64FC1, changeCoordArray);

	modelview = changeCoord*modelview;
}

GLfloat* convertMatrixType(const cv::Mat& m)
{
	typedef double precision;

	Size s = m.size();
	GLfloat* mGL = new GLfloat[s.width*s.height];

	for(int ix = 0; ix < s.width; ix++)
	{
		for(int iy = 0; iy < s.height; iy++)
		{
			mGL[ix*s.height + iy] = m.at<precision>(iy, ix);
		}
	}

	return mGL;
}

bool find_rough(Mat src, Point& object_center, Rect& object){
	int erosion_size = 3;
	int max_size=0, max_number=0;
	Mat element = getStructuringElement( MORPH_RECT, Size( 2*erosion_size + 1, 2*erosion_size+1 ), Point( erosion_size, erosion_size ) );
	Mat contour_mat;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	vector<RotatedRect> objects;
	getNormalizedRGB(src);
	/*
	cvtColor(src, src, CV_BGR2HSV);
	inRange(src, Scalar(hue_min,lum_min,sat_min), Scalar(hue_max,lum_max,sat_max), mask);
	morphologyEx( mask, mask , MORPH_CLOSE, element , Point(-1,-1), 1);
	*/
	int thresh = 180;
	for(int col=0; col<src.cols; col++){
		for(int row=0; row<src.rows; row++){
			if(find_euclidian(float(src.at<Vec3b>(row, col)[0]), float(src.at<Vec3b>(row, col)[1]), float(src.at<Vec3b>(row, col)[2]),0,255,0) < thresh){
				mask.at<uchar>(row, col)=255;
			}
			else
				mask.at<uchar>(row, col)=0;
		}

	}

	contour_mat = mask.clone();
	findContours( contour_mat, contours, hierarchy, CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE, Point(0, 0) );
	for(int i=0; i<contours.size(); i++){
		objects.push_back(minAreaRect(contours[i]));
		if(objects[i].size.area()>max_size){
			max_size = objects[i].size.area();
			max_number = i;
		}
	}
	if(contours.size()>0){
		if(objects[max_number].size.height>src.size().width/100){
			object_center = objects[max_number].center;
			radius = objects[max_number].size.height;
			if((objects[max_number].center.x+objects[max_number].size.width) < src.size().width 
				&& (objects[max_number].center.x-objects[max_number].size.width)>0
				&&(objects[max_number].center.y+objects[max_number].size.height) < src.size().height
				&&(objects[max_number].center.y-objects[max_number].size.height)>0)
			{
			object = Rect(Point(object_center.x+(objects[max_number].size.width*0.5), object_center.y+(objects[max_number].size.height*0.5)), 
				Point(object_center.x-(objects[max_number].size.width*0.5), object_center.y-(objects[max_number].size.height*0.5)));
			//cout<<"rectangle height "<<object.height<<" rectangle width "<<object.width<<endl;
			return true;
			}
		}
	}
	radius = 0;
	return false;
}

float find_euclidian(float r, float g, float b, float r_t, float g_t, float b_t){
	//D = sqrt((R-0)^2 + (G-255)^2 + (B-0)^2)
	float dist = (pow( r-r_t, 2) + pow(g-g_t, 2) + pow(b-b_t, 2));
	return pow(dist, 0.5);
}

Mat getNormalizedRGB(const Mat& rgb) {
	assert(rgb.type() == CV_8UC3);
	Mat rgb32f; rgb.convertTo(rgb32f, CV_32FC3);

	vector<Mat> split_rgb; split(rgb32f, split_rgb);
	Mat sum_rgb = split_rgb[0] + split_rgb[1] + split_rgb[2];
	split_rgb[0].setTo(0); //split_rgb[0] / sum_rgb;
	split_rgb[1] = split_rgb[1] / sum_rgb;
	split_rgb[2] = split_rgb[2] / sum_rgb;
	merge(split_rgb,rgb32f);
	return rgb32f;
	}

/*
gets the closest vertex in the model to the cursor location
Point pointerLoc : the center of the tracking circle 
*/
vec3 getClosest(Point pointerLoc)
{
	// change the image coordinate to a homogenous vec3
	float Z, X, Y;

	Z = (baseRadius/radius)*markerZvalue;

	//This takes the image coordinates of the green marker and converts them to world coordinates
	float x = pointerLoc.x - cameraMatrix.at<double>(0,2);
	float y = pointerLoc.y - cameraMatrix.at<double>(1,2);
	X = -(x*Z)/cameraMatrix.at<double>(0,0);
	Y = (y*Z)/cameraMatrix.at<double>(1,1);

	cout << "ModelView X:" << modelview.at<double>(0,3) << endl;
	cout << "ModelView Y:" << modelview.at<double>(1,3) << endl;
	cout << "ModelView Z:" << modelview.at<double>(2,3) << endl;
	cout << "Marker X:" << X << " Marker Y:" << Y << " Marker Z:" << Z << endl;

	vec3 pointerLocHomogenous = vec3(X, Y, Z);

	// get the image point in world space 
	worldPos = convertToModelCoords(pointerLocHomogenous);

	//cout << "this is where it is in world space ";
	//print(worldPos);
	// find the closest vertex 
	int closest = 0; 
	vec3 current;
	float currentDist;
	float closestDist = 500;
	for (int i=0; i<36; i++)
	{
		current.v[0] = vertices[3*i];
		current.v[1] = vertices[3*i+1];
		current.v[2] = vertices[3*i+2];

		currentDist = getDist(worldPos,current);
		if(currentDist < closestDist)
		{
			closestDist = currentDist;
			closest = i; 
		}
	}

	closestPoint = vec3(vertices[3*closest],vertices[3*closest+1],vertices[3*closest+2]);
	// return the index of the closest vertex 
	return closestPoint; 
}
/*
	gets the distance between two points.
*/
float getDist(vec3 point, vec3 otherPoint)
{
	float dist; 

	float xDiff = point.v[0] - otherPoint.v[0];
	float yDiff = point.v[1] - otherPoint.v[1];
	float zDiff = point.v[2] - otherPoint.v[2];

	dist = sqrtf((xDiff*xDiff)+(yDiff*yDiff)+(zDiff*zDiff));

	return dist; 
}