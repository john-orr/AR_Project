#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_DEPRECATE
#endif

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
GLfloat vertices[108];
GLuint numVertices;

//Calibration variables
bool calibrateZ = false;
double markerZvalue = -1;
double baseRadius = 1;
bool calibrated = false;
bool grabbed = false;

Size patternsize(7,7); //interior number of corners
Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
vector<Mat> rvecs, tvecs;
vector<vector<Point2f>> imagePoints;
vector<vector<Point3f>> objectPoints;
Mat projection = Mat::zeros(4, 4, CV_64F);;
Mat modelview = Mat::zeros(4, 4, CV_64F);;
Mat openGLtoCV;
mat4 model, view, persp_proj;
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

bool found=false;
Mat object_mat;
Rect obj;
int radius;
Mat mask;
Mat getNormalizedRGB(const Mat& rgb);
bool find_rough(Mat src, Point& object_center, Rect& object);
float find_euclidian(float r, float g, float b, float r_t, float g_t, float b_t);

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


#pragma endregion SHADER_FUNCTIONS

#pragma region VBO_FUNCTIONS
GLuint generateObjectBuffer(GLfloat vertices[], GLfloat colors[]) {
	numVertices = 36;
	// Genderate 1 generic buffer object, called VBO
	GLuint VBO;
 	glGenBuffers(1, &VBO);
	// In OpenGL, we bind (make active) the handle to a target name and then execute commands on that target
	// Buffer will contain an array of vertices 
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// After binding, we now fill our object with data, everything in "Vertices" goes to the GPU
	glBufferData(GL_ARRAY_BUFFER, numVertices*7*sizeof(GLfloat), NULL, GL_STREAM_DRAW);
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
	GLfloat* modelV = convertMatrixType(modelview);
	mat4 temp = mat4(modelV[0], modelV[4], modelV[8], modelV[12],
						modelV[1], modelV[5], modelV[9], modelV[13],
						modelV[2], modelV[6], modelV[10], modelV[14],
						modelV[3], modelV[7], modelV[11], modelV[15]);

	//derive inverse of the matrices
	mat4 model_inv = inverse(temp);
	mat4 view_inv = inverse(view);
	mat4 proj_inv = inverse(persp_proj);

	//undo the effects of those darn matrices
	vec4 vec4_worldcoords = vec4(worldcoords, 1); //is this even the right way to do it?
	vec3 result = model_inv*view_inv*proj_inv*vec4_worldcoords;

	return result;
}

void display(){

	cap >> frame;

	cap >> frame;
	Point object_pt;
	mask = Mat(frame.size(), CV_8UC1);
	if(found == false)
		found = find_rough(frame, object_pt, obj);
	if(found == true){
		circle(frame, object_pt, radius, CV_RGB(0,0,255), 1, 8, 0);
		found =false;
	}

	if(!calibrateZ)
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


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(!calibrated){
		calibrateCameraMatrix();
	} else{
		ChessBoard();
	}
	int selected_loc = glGetUniformLocation(shaderProgramID, "selected");

	glUniform3f(selected_loc,temp.v[0],temp.v[1],temp.v[2]);

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
	

	persp_proj = perspective(170.0, (float)frame.cols/(float)frame.rows, 0.1, 500.0);
	view = identity_mat4();
	model = translate(identity_mat4(),vec3(0,0,-0.1));

	glUniformMatrix4fv (proj_mat_location, 1, GL_FALSE, persp_proj.m);
	glUniformMatrix4fv (view_mat_location, 1, GL_FALSE, view.m);
	glUniformMatrix4fv (model_mat_location, 1, GL_FALSE, model.m);

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
}

void keypress(unsigned char key, int x, int y){
	if (key == 'g')
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
	}
}

int main(int argc, char** argv)
{
	ErrorTxt = fopen("error.txt","w");
	cap = VideoCapture(0);
	cap >> frame;
	cap >> frame;
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

	cvtColor(frame, gray, CV_BGR2GRAY); //source image

	patternfound = findChessboardCorners(gray, patternsize, points, CALIB_CB_FAST_CHECK);

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

		int proj_mat_location = glGetUniformLocation (shaderProgramID, "proj");
		int view_mat_location = glGetUniformLocation (shaderProgramID, "view");
		int model_mat_location = glGetUniformLocation (shaderProgramID, "model");

		GLfloat* modelV = convertMatrixType(modelview);
		GLfloat* projV = convertMatrixType(projection);

		if(!calibrateZ)
		{
			markerZvalue = modelview.at<double>(2,3);
			baseRadius = radius;
			calibrateZ = true;
		}

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
	float Z;
	if(radius !=0 && baseRadius != 0)
	{
		Z = -(radius/baseRadius)*markerZvalue;
	}else
	{
		Z = -1;
	}
	float X = -(pointerLoc.x*Z)/cameraMatrix.at<double>(0,0);
	float Y = (pointerLoc.y*Z)/cameraMatrix.at<double>(1,1);
	

	vec3 pointerLocHomogenous = vec3(X, Y, Z);//1.0f);
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