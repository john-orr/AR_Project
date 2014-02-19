#include <iostream>

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

//Calibration variables
bool calibrated = false;
Size patternsize(7,7); //interior number of corners
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
vector<Mat> rvecs, tvecs;
vector<vector<Point2f>> imagePoints;
vector<vector<Point3f>> objectPoints;
Mat projection = Mat::zeros(4, 4, CV_64F);;
Mat modelview = Mat::zeros(4, 4, CV_64F);;
Mat openGLtoCV;
mat4 model;
int testImages = 0;
double zNear = 0.1;
double zFar = 500;

void overlayImage();
void ChessBoard();
void calibrateCameraMatrix();
float* getTransform(Mat& mat);
void generateProjectionModelview(const cv::Mat& calibration, const cv::Mat& rotation, const cv::Mat& translation, cv::Mat& projection, cv::Mat& modelview);
GLfloat* convertMatrixType(const cv::Mat& m);

// Macro for indexing vertex buffer
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace std;


static const char* pVS = "                                                    \n\
#version 330                                                                  \n\
                                                                              \n\
in vec3 vPosition;															  \n\
in vec4 vColor;																  \n\
out vec4 color;																 \n\
uniform mat4 proj, view, model;                                                                              \n\
                                                                               \n\
void main()                                                                     \n\
{                                                                                \n\
    gl_Position = proj * view * model * vec4(vPosition.x, vPosition.y, vPosition.z, 1.0);  \n\
	color = vColor;							\n\
}";

static const char* pFS = "                                              \n\
#version 330                                                            \n\
                                                                        \n\
out vec4 FragColor;                                                      \n\
in vec4 color;                                                                          \n\
void main()                                                               \n\
{                                                                          \n\
FragColor = color;									 \n\
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
	GLuint numVertices = 36;
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
}
#pragma endregion VBO_FUNCTIONS


void display(){

	cap >> frame;

	perspective_warped_image = Mat::zeros(frame.rows, frame.cols, CV_8UC3);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
	if(!calibrated)
	{
		calibrateCameraMatrix();
	}

	if(calibrated)
	{
		ChessBoard();
	}

	glDrawArrays(GL_TRIANGLES, 0, 36);
	glutSwapBuffers();

	buffer = new unsigned char[frame.cols*frame.rows*3];
	glReadPixels(0, 0, frame.cols, frame.rows, GL_BGR, GL_UNSIGNED_BYTE, buffer);
	Mat image(frame.rows, frame.cols, CV_8UC3, buffer);
	flip(image,image,0);

	openGLtoCV = image.clone();
	
	overlayImage();
	imshow("Show Image", frame);

	glutPostRedisplay();
}


void init()
{	
	// Create 3 vertices that make up a triangle that fits on the viewport 
	GLfloat vertices[] = {  
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
	// Create a color array that identfies the colors of each vertex (format R, G, B, A)
	GLfloat colors[] = {0.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f,
						0.0f, 1.0f, 0.0f, 1.0f,
						1.0f, 0.0f, 0.0f, 1.0f,
						0.0f, 0.0f, 1.0f, 1.0f};
	// Set up the shaders
	shaderProgramID = CompileShaders();
	// Put the vertices and colors into a vertex buffer object
	generateObjectBuffer(vertices, colors);
	// Link the current buffer to the shader
	linkCurrentBuffertoShader(shaderProgramID);	
	glUseProgram (shaderProgramID);

	int proj_mat_location = glGetUniformLocation (shaderProgramID, "proj");
	int view_mat_location = glGetUniformLocation (shaderProgramID, "view");
	int model_mat_location = glGetUniformLocation (shaderProgramID, "model");

	mat4 persp_proj = perspective(170.0, (float)frame.cols/(float)frame.rows, 0.1, 500.0);
	mat4 view = identity_mat4();
	model = translate(identity_mat4(),vec3(0,0,-0.1));
	
	glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, persp_proj.m);
	glUniformMatrix4fv (view_mat_location, 1, GL_FALSE, view.m);
	glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, model.m);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
}

int main(int argc, char** argv)
{
	cap = VideoCapture(0);
	cap >> frame;
	cap >> frame;
	cout << "Width:" << frame.cols << " Height:" << frame.rows << endl;
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
	glViewport (0, 0, frame.cols, frame.rows);
	// Begin infinite event loop
	glutMainLoop();
    return 0;

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
		
		for(int i = 0; i<7; i++){
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
	
	}

	if(testImages >= 1)
	{
		calibrated = true;
		cout << "calibrating! No Images:" << testImages + 1 << endl;
		calibrateCamera(objectPoints, imagePoints, gray.size(), cameraMatrix, distCoeffs, rvecs, tvecs, 0);
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

	cvtColor(frame, gray, CV_BGR2GRAY); //source image
		
	//CALIB_CB_FAST_CHECK saves a lot of time on images
	//that do not contain any chessboard corners
	patternfound = findChessboardCorners(gray, patternsize, points, CALIB_CB_FAST_CHECK);

	if(patternfound)
	{

		//cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

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

		int proj_mat_location = glGetUniformLocation (shaderProgramID, "proj");
		int view_mat_location = glGetUniformLocation (shaderProgramID, "view");
		int model_mat_location = glGetUniformLocation (shaderProgramID, "model");

		GLfloat* modelV = convertMatrixType(modelview);
		GLfloat* projV = convertMatrixType(projection);
		

		glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, modelV);
		glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, projV);

	}

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