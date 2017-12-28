#include "main.h"

// Define the input array
float*** fMain_Arr;
float* fTemp_1D_Arr;
float** fTemp_2D_Arr;
float*** fTemp_3D_Arr;
// Define filters in each layer
float*** fConv_0;// 1 Channel-image-input
float*** fConv_1_dw;
float** fConv_1_pw;
float*** fConv_2_dw;
float** fConv_2_pw;
float*** fConv_3_dw;
float** fConv_3_pw;
float*** fConv_4_dw;
float** fConv_4_pw;
float*** fConv_5_dw;
float** fConv_5_pw;
float*** fConv_6_dw;
float** fConv_6_pw;
//Iteration
float*** fConv_7_dw;
float** fConv_7_pw;
float*** fConv_8_dw;
float** fConv_8_pw;
float*** fConv_9_dw;
float** fConv_9_pw;
float*** fConv_10_dw;
float** fConv_10_pw;
float*** fConv_11_dw;
float** fConv_11_pw;
//
float*** fConv_12_dw;
float** fConv_12_pw;
float*** fConv_13_dw;
float** fConv_13_pw;
// Fully connected layer
float** fFully_0;

// Allocate memories to each array
float* fUni_Arr(int iChannel);
float** fDu_Arr(int iChannel, int iNumber);
float*** fTri_Arr(int iChannel, int iWidth, int iHeight);
float**** fQuadri_Arr(int iChannel, int iWidth, int iHeight, int iNumber);

// Free memories to each array
void funcFree_Uni_Arr(float* fArr_Input);
void funcFree_Du_Arr(int iChannel, float** fArr_Input);
void funcFree_Tri_Arr(int iChannel, int iWidth, float*** fArr_Input);
void funcFree_Quadri_Arr(int iChannel, int iWidth, int iHeight, float**** fArr_Input);

// Load weights
void funcLoad_Weight_2D(int iChannel, int iNumber, float** fArr_Input, const char *cFilename[]);
void funcLoad_Weight_3D(int iChannel, int iWidth, int iHeight, float*** fArr_Input, const char *cFilename[]);

// Layer operation
void funcConv_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iStride_Num, float*** fArr_IMG, float*** fArr_Filter, float*** fOutput_IMG);
void funcDepth_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iStride_Num, float*** fArr_IMG, float*** fArr_Filter, float*** fOutput_IMG);
void fincPoint_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iNumber_Filter, float*** fArr_IMG, float** fArr_Filter, float*** fOutput_IMG);
void funcBatch_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fOutput_IMG, float fGamma, float fBeta);
void funcReLU_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG);
void funcPool_Average_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fArr_Filter);
void funcPool_Maximum_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fArr_Filter);
void funcFullyConnected_Operation(int iChannel_IMG, int iNum_Class, float*** fArr_IMG, float** fArr_Filter, float*** fOutput_IMG);

// Define load image files
RGBTRIPLE** Load_IMG(char * filename);
// Save images
void Save_IMG(char * filename, RGBTRIPLE** temp_ImageData);
// Show images
void Show_IMG(char * filename, RGBTRIPLE** temp_ImageData);
////////////////////
int iImage_Number_TEMP_TEMP = 0;
/////////////////

int main(int argc, char* argv[]) {	
	// Define exit button
	char cExitKey;

	//// Allocate memories to each array
	// Allocate the main input array with initialization
	fMain_Arr = fTri_Arr(iChannel_06*iAlpha_Width, iResolution_ininital, iResolution_ininital);
	
	// Allocate temp arraies
	//fTemp_1D_Arr = fUni_Arr(iChannel_06*iAlpha_Width);
	//fTemp_2D_Arr = fDu_Arr(iChannel_06*iChannel_06*iAlpha_Width*iAlpha_Width, iChannel_06*iChannel_06*iAlpha_Width*iAlpha_Width);
	fTemp_3D_Arr = fTri_Arr(iChannel_06*iAlpha_Width, iResolution_ininital, iResolution_ininital);

	// Allocate dynamic memories to each array
	// Allocate the convolutional filter
	fConv_0 = fTri_Arr(iChannel_01*iAlpha_Width, 3, 3); // 3 x 3 x 32

	// Allocate depth-wise convolutional filters
	fConv_1_dw = fTri_Arr(iChannel_01*iAlpha_Width, 3, 3); // 3 x 3 x 32 dw
	fConv_2_dw = fTri_Arr(iChannel_02*iAlpha_Width, 3, 3); // 3 x 3 x 64 dw
	fConv_3_dw = fTri_Arr(iChannel_03*iAlpha_Width, 3, 3); // 3 x 3 x 128 dw
	fConv_4_dw = fTri_Arr(iChannel_03*iAlpha_Width, 3, 3); // 3 x 3 x 128 dw
	fConv_5_dw = fTri_Arr(iChannel_04*iAlpha_Width, 3, 3); // 3 x 3 x 256 dw
	fConv_6_dw = fTri_Arr(iChannel_04*iAlpha_Width, 3, 3); // 3 x 3 x 256 dw
	// Iteration
	fConv_7_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw
	fConv_8_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw
	fConv_9_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw
	fConv_10_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw
	fConv_11_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw

	fConv_12_dw = fTri_Arr(iChannel_05*iAlpha_Width, 3, 3); // 3 x 3 x 512 dw
	fConv_13_dw = fTri_Arr(iChannel_06*iAlpha_Width, 3, 3); // 3 x 3 x 1024 dw
	
	// Allocate point-wise convolutional filters
	fConv_1_pw = fDu_Arr(iChannel_01*iAlpha_Width, iChannel_02*iAlpha_Width); // 1 x 1 x 32 x 64 pw
	fConv_2_pw = fDu_Arr(iChannel_02*iAlpha_Width, iChannel_03*iAlpha_Width); // 1 x 1 x 64 x 128 pw
	fConv_3_pw = fDu_Arr(iChannel_03*iAlpha_Width, iChannel_03*iAlpha_Width); // 1 x 1 x 128 x 128 pw
	fConv_4_pw = fDu_Arr(iChannel_03*iAlpha_Width, iChannel_04*iAlpha_Width); // 1 x 1 x 128 x 256 pw
	fConv_5_pw = fDu_Arr(iChannel_04*iAlpha_Width, iChannel_04*iAlpha_Width); // 1 x 1 x 256 x 256 pw
	fConv_6_pw = fDu_Arr(iChannel_04*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 256 x 512 pw
	// Iteration(same filter format)
	fConv_7_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
	fConv_8_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
	fConv_9_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
	fConv_10_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
	fConv_11_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw

	fConv_12_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_06*iAlpha_Width); // 1 x 1 x 512 x 1024 pw
	fConv_13_pw = fDu_Arr(iChannel_06*iAlpha_Width, iChannel_06*iAlpha_Width); // 1 x 1 x 1024 x 1024 pw

	fFully_0 = fDu_Arr(iChannel_06*iAlpha_Width, iResult_Class);
	//// End allocation process

		 
	//// Load trained weights from '.txt' files	
	const char *cCharName_TEMP[100];
	//int iLoad_Num = 0;
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_01*iAlpha_Width, 3, 3, fConv_0, cCharName_TEMP);
	//
	//iLoad_Num = 0;
	////1
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_01*iAlpha_Width, 3, 3, fConv_1_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_01*iAlpha_Width, iChannel_02*iAlpha_Width, fConv_1_pw, cCharName_TEMP);
	//iLoad_Num++;
	//
	////2
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_02*iAlpha_Width, 3, 3, fConv_2_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_02*iAlpha_Width, iChannel_03*iAlpha_Width, fConv_2_pw, cCharName_TEMP);
	//iLoad_Num++;

	////3
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_03*iAlpha_Width, 3, 3, fConv_3_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_03*iAlpha_Width, iChannel_03*iAlpha_Width, fConv_3_pw, cCharName_TEMP);
	//iLoad_Num++;

	////4
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_03*iAlpha_Width, 3, 3, fConv_4_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_03*iAlpha_Width, iChannel_04*iAlpha_Width, fConv_4_pw, cCharName_TEMP);
	//iLoad_Num++;

	////5
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_04*iAlpha_Width, 3, 3, fConv_5_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_04*iAlpha_Width, iChannel_04*iAlpha_Width, fConv_5_pw, cCharName_TEMP);
	//iLoad_Num++;

	////6
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_04*iAlpha_Width, 3, 3, fConv_6_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_04*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_6_pw, cCharName_TEMP);
	//iLoad_Num++;

	////7(iteration)
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_7_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_7_pw, cCharName_TEMP);
	//iLoad_Num++;

	////8
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_8_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_8_pw, cCharName_TEMP);
	//iLoad_Num++;

	////9
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_9_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_9_pw, cCharName_TEMP);
	//iLoad_Num++;

	////10
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_10_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_10_pw, cCharName_TEMP);
	//iLoad_Num++;

	////11(end iteration)
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_11_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width, fConv_11_pw, cCharName_TEMP);
	//iLoad_Num++;

	////12
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_05*iAlpha_Width, 3, 3, fConv_12_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_05*iAlpha_Width, iChannel_06*iAlpha_Width, fConv_12_pw, cCharName_TEMP);
	//iLoad_Num++;

	////13
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_dw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_3D(iChannel_06*iAlpha_Width, 3, 3, fConv_13_dw, cCharName_TEMP);
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_pw_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_06*iAlpha_Width, iChannel_06*iAlpha_Width, fConv_13_pw, cCharName_TEMP);
	//iLoad_Num++;

	//iLoad_Num = 0;
	//sprintf_s(cCharName_TEMP, sizeof(cCharName_TEMP), "conv_fc_%s.txt", iLoad_Num);// Relative path
	//funcLoad_Weight_2D(iChannel_06*iAlpha_Width, iResult_Class, fFully_0, cCharName_TEMP);
	////// End load process

	int iImage_Number_TEMP = 0;
	char cImage_name[100];
	RGBTRIPLE** rINPUT_IMG;

	//// Convolutional operation during the process
	while (1) {
		// EXIT process
		// If an user press any button, it would be true
		if (_kbhit())
		{
			// If the pressed button is any control button, "getc()" would be false
			if (!(cExitKey = _getch()))
			{
				cExitKey = _getch();
			}
			// If the pressed button is "ESC" button, the char would be true and exit "while(1)"
			if (cExitKey == 27)// "char cExitKey == 27" means ESC
			{
				printf("You press the END button!\n");
				break;
			}
		}//End EXIT process

		// Image show
		sprintf_s(cImage_name, sizeof(cImage_name), "%d.bmp", iImage_Number_TEMP);
		//printf("%s Load\n", cImage_name);
		rINPUT_IMG = Load_IMG(cImage_name);
		Show_IMG(cImage_name, rINPUT_IMG);
		iImage_Number_TEMP = iImage_Number_TEMP + 1;

		clock_t before;
		double result;
		before = clock();


		// Main while processing
		funcConv_Operation(1, iResolution_ininital, iResolution_ininital, 2, fMain_Arr, fConv_0, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr);
		
		//1
		funcDepth_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, 1, fMain_Arr, fConv_1_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr);
		fincPoint_Operation(iChannel_01*iAlpha_Width, iResolution_01, iResolution_01, iChannel_02*iAlpha_Width, fMain_Arr, fConv_1_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_02*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_02*iAlpha_Width, iResolution_01, iResolution_01, fMain_Arr);
		
		//2
		funcDepth_Operation(iChannel_02*iAlpha_Width, iResolution_01, iResolution_01, 2, fMain_Arr, fConv_2_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_02*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_02*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr);
		fincPoint_Operation(iChannel_02*iAlpha_Width, iResolution_02, iResolution_02, iChannel_03*iAlpha_Width, fMain_Arr, fConv_2_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr);
		
		//3
		funcDepth_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, 1, fMain_Arr, fConv_3_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr);
		fincPoint_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, iChannel_03*iAlpha_Width, fMain_Arr, fConv_3_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, fMain_Arr);

		//4
		funcDepth_Operation(iChannel_03*iAlpha_Width, iResolution_02, iResolution_02, 2, fMain_Arr, fConv_4_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_03*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_03*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr);
		fincPoint_Operation(iChannel_03*iAlpha_Width, iResolution_03, iResolution_03, iChannel_04*iAlpha_Width, fMain_Arr, fConv_4_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr);

		//5
		funcDepth_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, 1, fMain_Arr, fConv_5_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr);
		fincPoint_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, iChannel_04*iAlpha_Width, fMain_Arr, fConv_5_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, fMain_Arr);

		//6
		funcDepth_Operation(iChannel_04*iAlpha_Width, iResolution_03, iResolution_03, 2, fMain_Arr, fConv_6_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_04*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_04*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_04*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_6_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);


		//7 (iteration start)
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 1, fMain_Arr, fConv_7_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_7_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);

		//8
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 1, fMain_Arr, fConv_8_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_8_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);

		//9
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 1, fMain_Arr, fConv_9_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_9_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);

		//10
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 1, fMain_Arr, fConv_10_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_10_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);

		//11 (iteration end)
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 1, fMain_Arr, fConv_11_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, iChannel_05*iAlpha_Width, fMain_Arr, fConv_11_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, fMain_Arr);

		//12
		funcDepth_Operation(iChannel_05*iAlpha_Width, iResolution_04, iResolution_04, 2, fMain_Arr, fConv_12_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_05*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_05*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr);
		fincPoint_Operation(iChannel_05*iAlpha_Width, iResolution_05, iResolution_05, iChannel_06*iAlpha_Width, fMain_Arr, fConv_12_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr);

		//13 (paper error: stride 2 (original) -> stride 1 (actual)
		funcDepth_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, 1, fMain_Arr, fConv_13_dw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr);
		fincPoint_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, iChannel_06*iAlpha_Width, fMain_Arr, fConv_13_pw, fTemp_3D_Arr);
		funcBatch_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr, fTemp_3D_Arr, fGamma_Batch, fBeta_Batch);
		funcReLU_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr);

		//Average pooling process
		funcPool_Average_Operation(iChannel_06*iAlpha_Width, iResolution_05, iResolution_05, fMain_Arr, fTemp_3D_Arr);

		//Fully connected process
		funcFullyConnected_Operation(iChannel_06*iAlpha_Width, iResult_Class, fMain_Arr, fFully_0, fTemp_3D_Arr);


		// Results
		if (fMain_Arr[0][0][0] >= fMain_Arr[1][0][0]) {
			printf("Driver, ");
			//printf("%d.bmp: Driver\n", iImage_Number_TEMP - 1);
		}
		else {
			printf("Non-driver, ");
			//printf("%d.bmp: Non-driver \n", iImage_Number_TEMP - 1);
		}
		result = (double)(clock() - before) / CLOCKS_PER_SEC;
		printf("Consume %5.2f sec\n\n", result);

		//printf("OneFrameEnd \n\n");
	}
	// Memory release
	funcFree_Tri_Arr(iChannel_06*iAlpha_Width, iResolution_ininital, fMain_Arr);
	funcFree_Tri_Arr(iChannel_06*iAlpha_Width, iResolution_ininital, fTemp_3D_Arr);

	funcFree_Tri_Arr(iChannel_01*iAlpha_Width, 3, fConv_0);
	funcFree_Tri_Arr(iChannel_01*iAlpha_Width, 3, fConv_1_dw);
	funcFree_Tri_Arr(iChannel_02*iAlpha_Width, 3, fConv_2_dw);
	funcFree_Tri_Arr(iChannel_03*iAlpha_Width, 3, fConv_3_dw);
	funcFree_Tri_Arr(iChannel_03*iAlpha_Width, 3, fConv_4_dw);
	funcFree_Tri_Arr(iChannel_04*iAlpha_Width, 3, fConv_5_dw);
	funcFree_Tri_Arr(iChannel_04*iAlpha_Width, 3, fConv_6_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_7_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_8_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_9_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_10_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_11_dw);
	funcFree_Tri_Arr(iChannel_05*iAlpha_Width, 3, fConv_12_dw);
	funcFree_Tri_Arr(iChannel_06*iAlpha_Width, 3, fConv_13_dw);

	funcFree_Du_Arr(iChannel_01*iAlpha_Width, fConv_1_pw);
	funcFree_Du_Arr(iChannel_02*iAlpha_Width, fConv_2_pw);
	funcFree_Du_Arr(iChannel_03*iAlpha_Width, fConv_3_pw);
	funcFree_Du_Arr(iChannel_03*iAlpha_Width, fConv_4_pw);
	funcFree_Du_Arr(iChannel_04*iAlpha_Width, fConv_5_pw);
	funcFree_Du_Arr(iChannel_04*iAlpha_Width, fConv_6_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_7_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_8_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_9_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_10_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_11_pw);
	funcFree_Du_Arr(iChannel_05*iAlpha_Width, fConv_12_pw);
	funcFree_Du_Arr(iChannel_06*iAlpha_Width, fConv_13_pw);

	funcFree_Du_Arr(iChannel_06*iAlpha_Width, fFully_0);

	printf("Program End\n\n");

	return EXIT_SUCCESS;
}

// Allocate memories to the dimmension of each array
float* fUni_Arr(int iChannel)
{
	int i;
	float* ptr;

	if ((ptr = (float*)malloc(iChannel * sizeof(float))) == NULL)
	{
		printf("\nMemory allocation failure\n");
		exit(1);
	}

	// Initialization
	for (i = 0; i < iChannel; i++) {
			ptr[i] = 0;
	}

	printf("\nMEMORY ALLOCATION(float) OK!\n");
	return ptr;
}

float** fDu_Arr(int iChannel, int iNumber)
{
	int i, j;
	float** ptr;

	if ((ptr = (float**)malloc(iChannel * sizeof(float*))) == NULL)
	{
		printf("\nMemory allocation failure\n");
		exit(1);
	}

	for (i = 0; i < iChannel; i++) {
		if ((ptr[i] = (float*)malloc(iNumber * sizeof(float))) == NULL)
		{
			printf("\nMemory allocation failure\n");
			exit(1);
		}
	}

	// Initialization
	for (i = 0; i < iChannel; i++) {
		for (j = 0; j < iNumber; j++) {
			ptr[i][j] = 0;
		}
	}

	printf("\nMEMORY ALLOCATION(float) OK!\n");
	return ptr;
}

float*** fTri_Arr(int iChannel, int iWidth, int iHeight)
{
	int i, j, k;
	float*** ptr;

	if ((ptr = (float***)malloc(iChannel * sizeof(float**))) == NULL)
	{
		printf("\nMemory allocation failure\n");
		exit(1);
	}

	for (i = 0; i < iChannel; i++) {
		if ((ptr[i] = (float**)malloc(iWidth * sizeof(float*))) == NULL)
		{
			printf("\nMemory allocation failure\n");
			exit(1);
		}
	}

	for (i = 0; i < iChannel; i++){
		for (j = 0; j < iWidth; j++) {
			if ((ptr[i][j] = (float*)malloc(iHeight * sizeof(float))) == NULL)
			{
				printf("\nMemory allocation failure\n");
				exit(1);
			}
		}
	}

	// Initialization
	for (i = 0; i < iChannel; i++) {
		for (j = 0; j < iWidth; j++) {
			for (k = 0; k < iHeight; k++) {
				ptr[i][j][k] = 0;
			}
		}
	}

	printf("\nMEMORY ALLOCATION(float) OK!\n");
	return ptr;
}

float**** fQuadri_Arr(int iChannel, int iWidth, int iHeight, int iNumber)
{
	int i, j, k, l;
	float**** ptr;

	if ((ptr = (float****)malloc(iChannel * sizeof(float***))) == NULL)
	{
		printf("\nMemory allocation failure\n");
		exit(1);
	}

	for (i = 0; i < iChannel; i++) {
		if ((ptr[i] = (float***)malloc(iWidth * sizeof(float**))) == NULL)
		{
			printf("\nMemory allocation failure\n");
			exit(1);
		}
	}

	for (i = 0; i < iChannel; i++) {
		for (j = 0; j < iWidth; j++) {
			if ((ptr[i][j] = (float**)malloc(iHeight * sizeof(float*))) == NULL)
			{
				printf("\nMemory allocation failure\n");
				exit(1);
			}
		}
	}

	for (i = 0; i < iChannel; i++) {
		for (j = 0; j < iWidth; j++) {
			for (k = 0; k < iHeight; k++) {
				if ((ptr[i][j][k] = (float*)malloc(iNumber * sizeof(float))) == NULL)
				{
					printf("\nMemory allocation failure\n");
					exit(1);
				}
			}
		}
	}

	// Initialization
	for (i = 0; i < iChannel; i++) {
		for (j = 0; j < iWidth; j++) {
			for (k = 0; k < iHeight; k++) {
				for (l = 0; l < iNumber; l++) {
					ptr[i][j][k][l] = 0;
				}
			}
		}
	}

	printf("\nMEMORY ALLOCATION(float) OK!\n");
	return ptr;
}

// Free arrays
void funcFree_Uni_Arr(float* fArr_Input)
{
	free(fArr_Input);
	printf("\nMEMORY FREE(float) OK!\n");
	return fArr_Input;
}

void funcFree_Du_Arr(int iChannel, float** fArr_Input)
{
	int i;
	for (i = 0; i<iChannel; i++)
	{		
		free(fArr_Input[i]);
	}
	free(fArr_Input);
	printf("\nMEMORY FREE(float) OK!\n");

	return fArr_Input;
}

void funcFree_Tri_Arr(int iChannel, int iWidth, float*** fArr_Input)
{
	int i, j;
	for (i = 0; i<iChannel; i++)
	{
		for (j = 0; j<iWidth; j++)
		{
			free(fArr_Input[i][j]);
		}
		free(fArr_Input[i]);
	}
	free(fArr_Input);
	printf("\nMEMORY FREE(float) OK!\n");

	return fArr_Input;
}

void funcFree_Quadri_Arr(int iChannel, int iWidth, int iHeight, float**** fArr_Input)
{
	int i, j, k;
	for (i = 0; i<iChannel; i++)
	{
		for (j = 0; j<iWidth; j++)
		{
			for (k = 0; k < iHeight; k++) {
				free(fArr_Input[i][j][k]);
			}
			free(fArr_Input[i][j]);
		}
		free(fArr_Input[i]);
	}
	free(fArr_Input);
	printf("\nMEMORY FREE(float) OK!\n");

	return fArr_Input;
}


// Load File to 2D-array
void funcLoad_Weight_2D(int iChannel, int iNumber, float** fArr_Input, const char *cFilename[]) {

	FILE *infile = fopen(cFilename, "rb");

	if (!infile) {
		printf("ERROR: File %s could not be opened! \n", cFilename);
		exit(-1);
	}

	float* farrSerial_Information;
	farrSerial_Information = fUni_Arr(iChannel*iNumber);
	fread(farrSerial_Information, sizeof(float), iChannel*iNumber, infile);
	fclose(infile);

	int iLayer_Temp = 0;
	int iNumber_Temp = 0;
	long int iIndex_Temp = 0;

	for (iLayer_Temp = 0; iLayer_Temp < iChannel; iLayer_Temp++) {
		for (iNumber_Temp = 0; iNumber_Temp < iNumber; iNumber_Temp++) {
				fArr_Input[iLayer_Temp][iNumber_Temp] = (float)farrSerial_Information[iIndex_Temp];
				printf("%f\n", fArr_Input[iLayer_Temp][iNumber_Temp]);
				iIndex_Temp++;			
		}
	}
	return fArr_Input;
}

// Load File to 3D-array
void funcLoad_Weight_3D(int iChannel, int iWidth, int iHeight, float*** fArr_Input, const char *cFilename[]) {

	FILE *infile = fopen(cFilename, "rb");
	
	if (!infile) {
		printf("ERROR: File %s could not be opened! \n", cFilename);
		exit(-1);
	}

	float* farrSerial_Information;
	farrSerial_Information = fUni_Arr(iChannel*iWidth*iHeight);
	fread(farrSerial_Information, sizeof(float), iChannel*iWidth*iHeight, infile);
	fclose(infile);

	int iLayer_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long int iIndex_Temp = 0;

	for (iLayer_Temp = 0; iLayer_Temp < iChannel; iLayer_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth; iWidth_Temp++) {
			for (iHeight_Temp = 0; iHeight_Temp < iHeight; iHeight_Temp++) {
				fArr_Input[iLayer_Temp][iWidth_Temp][iHeight_Temp] = (float)farrSerial_Information[iIndex_Temp];
				printf("%f\n", fArr_Input[iLayer_Temp][iWidth_Temp][iHeight_Temp]);
				iIndex_Temp++;
			}
		}
	}
	return fArr_Input;
}


// Convolutional operation for Gray image and only use first conv layer
// It should change when it is general convolution layer because that has 4 dimmensions
void funcConv_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iStride_Num, float*** fArr_IMG, float*** fArr_Filter, float*** fOutput_IMG) {

	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long float iSumConv = 0;

	int iHEIGHT_HERE = 0;
	int iWIDTH_HERE = 0;

	if (iStride_Num == 1) {
		iHEIGHT_HERE = iHeight_IMG;
		iWIDTH_HERE = iWidth_IMG;
		// Convolution process
		for (iHeight_Temp = 0; iHeight_Temp < iHEIGHT_HERE; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWIDTH_HERE; iWidth_Temp++) {
				for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {

					if (iHeight_Temp == 0) {
						if (iWidth_Temp == 0) {
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
							fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
						}
						else {
							if (iWidth_Temp == iWidth_IMG - 1) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
						}
					}
					else {
						if (iHeight_Temp == iHeight_IMG - 1) {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;

							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {

									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
						else {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
					}
					iSumConv = 0;
				}//
			}
		}
	}
	else if (iStride_Num <= 0) {
		printf("Stride number is wrong\n");
		printf("Check out your code\n");
	}

	else {
		iHEIGHT_HERE = iHeight_IMG / iStride_Num;
		iWIDTH_HERE = iWidth_IMG / iStride_Num;
		// Convolution process
		for (iHeight_Temp = 0; iHeight_Temp < iHEIGHT_HERE; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWIDTH_HERE; iWidth_Temp++) {
				for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {

					if (iHeight_Temp == 0) {
						if (iWidth_Temp == 0) {
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
							fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
						}
						else {
							if (iWidth_Temp == iWidth_IMG - 1) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
						}
					}
					else {
						if (iHeight_Temp == iHeight_IMG - 1) {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;

							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {

									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
						else {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num* + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
					}
					iSumConv = 0;
				}//
			}
		}
	}

	
	
	// Copy results to the input array
	for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp];
			}
		}
	}

	return fArr_IMG;
}

// Depth wise convolutional operation
void funcDepth_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iStride_Num, float*** fArr_IMG, float*** fArr_Filter, float*** fOutput_IMG) {
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long float iSumConv = 0;

	int iHEIGHT_HERE = 0;
	int iWIDTH_HERE = 0;

	if (iStride_Num == 1) {
		iHEIGHT_HERE = iHeight_IMG;
		iWIDTH_HERE = iWidth_IMG;
		// Convolution process
		for (iHeight_Temp = 0; iHeight_Temp < iHEIGHT_HERE; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWIDTH_HERE; iWidth_Temp++) {
				for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {

					if (iHeight_Temp == 0) {
						if (iWidth_Temp == 0) {
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
							fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
						}
						else {
							if (iWidth_Temp == iWidth_IMG - 1) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
						}
					}
					else {
						if (iHeight_Temp == iHeight_IMG - 1) {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;

							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {

									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
						else {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp + 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
					}
					iSumConv = 0;
				}//
			}
		}
	}
	else if (iStride_Num <= 0) {
		printf("Stride number is wrong\n");
		printf("Check out your code\n");
	}

	else {
		iHEIGHT_HERE = iHeight_IMG / iStride_Num;
		iWIDTH_HERE = iWidth_IMG / iStride_Num;
		// Convolution process
		for (iHeight_Temp = 0; iHeight_Temp < iHEIGHT_HERE; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWIDTH_HERE; iWidth_Temp++) {
				for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {

					if (iHeight_Temp == 0) {
						if (iWidth_Temp == 0) {
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
							iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
							fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
						}
						else {
							if (iWidth_Temp == iWidth_IMG - 1) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
						}
					}
					else {
						if (iHeight_Temp == iHeight_IMG - 1) {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;

							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {

									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
						else {
							if (iWidth_Temp == 0) {
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num* +1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
								iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
								fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
							}
							else {
								if (iWidth_Temp == iWidth_IMG - 1) {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp - 1][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
								else {
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][0][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][1][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp - 1])*(fArr_Filter[iFilter_Temp][2][0]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][0][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][1][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp])*(fArr_Filter[iFilter_Temp][2][1]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp - 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][0][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][1][2]);
									iSumConv = iSumConv + (fArr_IMG[iFilter_Temp][iStride_Num*iWidth_Temp + 1][iStride_Num*iHeight_Temp + 1])*(fArr_Filter[iFilter_Temp][2][2]);
									fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
								}
							}
						}
					}
					iSumConv = 0;
				}//
			}
		}
	}

	// Copy results to the input array
	for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] = fOutput_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp];
			}
		}
	}

	return fArr_IMG;
}

// Point wise convolutional operation
void fincPoint_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, int iNumber_Filter, float*** fArr_IMG, float** fArr_Filter, float*** fOutput_IMG) {
	int iNumber_Temp = 0;
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long float iSumConv = 0;

	// Convolution process
	for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
			for (iNumber_Temp = 0; iNumber_Temp < iNumber_Filter; iNumber_Temp++) {
				for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
					iSumConv += (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp])*(fArr_Filter[iFilter_Temp][iNumber_Temp]);
				}
				fOutput_IMG[iNumber_Temp][iWidth_Temp][iHeight_Temp] = iSumConv;
				iSumConv = 0;
			}
		}
	}

	// Copy results to the input array
	for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
			for (iNumber_Temp = 0; iNumber_Temp < iNumber_Filter; iNumber_Temp++) {
				fArr_IMG[iNumber_Temp][iWidth_Temp][iHeight_Temp] = fOutput_IMG[iNumber_Temp][iWidth_Temp][iHeight_Temp];
			}
		}
	}

	return fArr_IMG;


}


// Batch normalization operation
void funcBatch_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fOutput_IMG, float fGamma, float fBeta) {
	
	// Sum of pixels
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long float fSumConv = 0;
	float fMeanConv = 0;
	float fVarConv = 0;
	long int iNum_Total = 0; // Total pixels in each channel
	iNum_Total = iHeight_IMG*iWidth_IMG;

	// Mean value process
	for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
		for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
				// Sum of pixels
				fSumConv += fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp];
			}
		}
		// Mean value
		fOutput_IMG[iFilter_Temp][0][0] = fSumConv / iNum_Total;
	}

	// Variance value process
	for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
		for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
				fSumConv += ((abs(fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] - fOutput_IMG[iFilter_Temp][0][0]))^2);
			}
		}
		// Variance value
		fOutput_IMG[iFilter_Temp][1][0] = fSumConv / iNum_Total;
	}
	
	// Normalization process + scale & shift process
	if (fGamma == 0) {
		if (fBeta == 0) {
			// Normalization process + scale & shift process
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
					for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
						// Sum of pixels
						fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]
							= ((fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] - fOutput_IMG[iFilter_Temp][0][0]) / (fOutput_IMG[iFilter_Temp][1][0] + 0.001));
					}
				}
			}
		}
		else {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
					for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
						// Sum of pixels
						fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]
							= ((fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] - fOutput_IMG[iFilter_Temp][0][0]) / (fOutput_IMG[iFilter_Temp][1][0] + 0.001)) + fBeta;
					}
				}
			}
		}		
	}
	else {
		if (fBeta == 0) {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
					for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
						// Sum of pixels
						fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]
							= ((fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] - fOutput_IMG[iFilter_Temp][0][0]) / (fOutput_IMG[iFilter_Temp][1][0] + 0.001))* fGamma;
					}
				}
			}		
		}
		else {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
					for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
						// Sum of pixels
						fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]
							= ((fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp] - fOutput_IMG[iFilter_Temp][0][0]) / (fOutput_IMG[iFilter_Temp][1][0] + 0.001))* fGamma + fBeta;
					}
				}
			}
		}
	}

	return fArr_IMG;
}

// ReLU operation
void funcReLU_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG) {
	// Variables for a convolution process
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	// ReLU process
	for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
		for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG; iWidth_Temp++) {
				fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]= max(fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp], 0.0);
			}
		}
	}
	return fArr_IMG;
}


// Average pooling operation
void funcPool_Average_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fArr_Filter) {
	
	// Variables for a convolution process
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	long float fPixel_SUM = 0;

	// Convolution process
	for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
		for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG ; iHeight_Temp++) {
			for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG ; iWidth_Temp++) {
				fPixel_SUM += (fArr_IMG[iFilter_Temp][iWidth_Temp][iHeight_Temp]);
			}
		}
		fArr_IMG[iFilter_Temp][0][0] = fPixel_SUM;
		fPixel_SUM = 0;
	}

	return fArr_IMG;
}

// Max-pooling operation
void funcPool_Maximum_Operation(int iChannel_IMG, int iWidth_IMG, int iHeight_IMG, float*** fArr_IMG, float*** fArr_Filter) {

	// Variables for a convolution process
	int iFilter_Temp = 0;
	int iHeight_Temp = 0;
	int iWidth_Temp = 0;
	float iPixel_1 = 0;
	float iPixel_2 = 0;
	float iPixel_3 = 0;
	float iPixel_4 = 0;

	// Convolution process
	for (iHeight_Temp = 0; iHeight_Temp < iHeight_IMG / 2; iHeight_Temp++) {
		for (iWidth_Temp = 0; iWidth_Temp < iWidth_IMG / 2; iWidth_Temp++) {
			for (iFilter_Temp = 0; iFilter_Temp < iChannel_IMG; iFilter_Temp++) {
				iPixel_1 = (fArr_IMG[iFilter_Temp][2 * iWidth_Temp][2 * iHeight_Temp]);
				iPixel_2 = (fArr_IMG[iFilter_Temp][2 * iWidth_Temp + 1][2 * iHeight_Temp]);
				iPixel_3 = (fArr_IMG[iFilter_Temp][2 * iWidth_Temp][2 * iHeight_Temp + 1]);
				iPixel_4 = (fArr_IMG[iFilter_Temp][2 * iWidth_Temp + 1][2 * iHeight_Temp + 1]);
				fArr_IMG[(2 * iFilter_Temp)][iWidth_Temp][iHeight_Temp] = max(iPixel_1, iPixel_2, iPixel_3, iPixel_4);
			}
		}
	}
	return fArr_Filter;
}


// Fully connected layer operation
void funcFullyConnected_Operation(int iChannel_IMG, int iNum_Class, float*** fArr_IMG, float** fArr_Filter, float*** fOutput_IMG) {
	int iIndex = 0;
	int iOutput_Index = 0;
	double dSum_of_Components = 0;

	for (iOutput_Index = 0; iOutput_Index < iNum_Class; iOutput_Index++) {
		dSum_of_Components = 0;
		for (iIndex = 0; iIndex < iChannel_IMG; iIndex++) {
			dSum_of_Components = dSum_of_Components + fArr_IMG[iIndex][0][0] * fArr_Filter[iIndex][iOutput_Index];
			iIndex++;
		}
		fOutput_IMG[iOutput_Index][0][0] = dSum_of_Components;
	}
	for (iOutput_Index = 0; iOutput_Index < iNum_Class; iOutput_Index++) {
		fArr_IMG [iOutput_Index][0][0] = fOutput_IMG[iOutput_Index][0][0];
	}
	return fArr_IMG;
}


// Image load process
RGBTRIPLE** Load_IMG(char * filename)
{
	FILE				*InData;
	int					i = 0, j = 0;
	RGBTRIPLE**			RGBIMG;
	RGBIMG = (RGBTRIPLE **)malloc(sizeof(RGBTRIPLE*)*Y_Size);
	for (int i = 0; i<Y_Size; i++) {
		RGBIMG[i] = (RGBTRIPLE*)malloc(sizeof(RGBTRIPLE) * X_Size);
	}

	// 파일 불러오기 //
	InData = fopen(filename, "rb");	

	if (InData == NULL)
	{
		printf("\nError: there is no file\n");
		fclose(InData);
		return NULL;
	}

	// 파일 정보 받기 // 비트맵 이미지인가? //

	fread(&Header_File, sizeof(BITMAPFILEHEADER), 1, InData);

	if (Header_File.bfType != 0x4d42)
	{
		printf("\n *** This file is not BMP *** \n");
		fclose(InData);
		return NULL;
	}

	// 이미지 정보 받기  //

	fread(&Header_Info, sizeof(BITMAPINFOHEADER), 1, InData);

	// Palette에 정보 입력 //
	//fread(ImageData, sizeof(unsigned char), Header_Info.biSizeImage, InData);
	fseek(InData, Header_File.bfOffBits, SEEK_SET);
	unsigned char temptemp;

	for (i = 0; i < Y_Size; i++)
	{
		for (j = 0; j < X_Size; j++)
		{
			fread(&(RGBIMG[i][j].rgbtRed), sizeof(char), 1, InData);		// Dynamic Allocation 이용하여 수정
			fread(&(RGBIMG[i][j].rgbtBlue), sizeof(char), 1, InData);
			fread(&(RGBIMG[i][j].rgbtGreen), sizeof(char), 1, InData);
			temptemp = &RGBIMG[i][j].rgbtRed;
			temptemp = &RGBIMG[i][j].rgbtBlue;
			temptemp = &RGBIMG[i][j].rgbtGreen;
		}
	}
	fclose(InData);
	return RGBIMG;
}

void Save_IMG(char * filename, RGBTRIPLE** temp_ImageData)
{
	int i, j;

	FILE *OutData = fopen(filename, "wb");

	fwrite(&Header_File, sizeof(char), sizeof(BITMAPFILEHEADER), OutData);
	fwrite(&Header_Info, sizeof(char), sizeof(BITMAPINFOHEADER), OutData);

	for (i = 0; i < Y_Size; i++)
	{
		for (j = 0; j < X_Size; j++)
		{
			fwrite(&(temp_ImageData[i][j].rgbtRed), sizeof(char), 1, OutData);
			fwrite(&(temp_ImageData[i][j].rgbtBlue), sizeof(char), 1, OutData);
			fwrite(&(temp_ImageData[i][j].rgbtGreen), sizeof(char), 1, OutData);
		}
	}

	fclose(OutData);
	return;
}

void Show_IMG(char * filename, RGBTRIPLE** temp_ImageData)
{
	Save_IMG(filename, temp_ImageData);

	HINSTANCE hInstance = GetModuleHandle(NULL);

	HWND hWnd = FindWindow("ConsoleWindowClass", NULL);

	HBITMAP hImage, hOldBitmap;




	HDC hdc = GetDC(hWnd);

	HDC hMemDC = CreateCompatibleDC(hdc);



	// 이미지 로드

	hImage = (HBITMAP)LoadImage(NULL, TEXT(filename), IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE | LR_CREATEDIBSECTION);

	// 이미지 출력 부분

	//while (1) {
		hOldBitmap = (HBITMAP)SelectObject(hMemDC, hImage);

		BitBlt(hdc, 0, 0, X_Size, Y_Size, hMemDC, 0, 0, SRCCOPY);
	//}

	// 각종 메모리 해제 

	SelectObject(hMemDC, hOldBitmap);

	DeleteObject(hImage);

	DeleteDC(hMemDC);

	ReleaseDC(hWnd, hdc);

	//system("pause");

	return;

}