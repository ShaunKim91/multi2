//#include "main.h"
//
//// Define filters in each layer
//float**** fConv_0;
//float*** fConv_1_dw;
//float** fConv_1_pw;
//float*** fConv_2_dw;
//float** fConv_2_pw;
//float*** fConv_3_dw;
//float** fConv_3_pw;
//float*** fConv_4_dw;
//float** fConv_4_pw;
//float*** fConv_5_dw;
//float** fConv_5_pw;
//float*** fConv_6_dw;
//float** fConv_6_pw;
////Iteration
//float*** fConv_7_dw;
//float** fConv_7_pw;
//float*** fConv_8_dw;
//float** fConv_8_pw;
//float*** fConv_9_dw;
//float** fConv_9_pw;
//float*** fConv_10_dw;
//float** fConv_10_pw;
//float*** fConv_11_dw;
//float** fConv_11_pw;
////
//float*** fConv_12_dw;
//float** fConv_12_pw;
//float*** fConv_13_dw;
//float** fConv_13_pw;
//// Fully connected layer
//float** fFully_0;
//
//float* fUni_Arr(int iChannel);
//float** fDu_Arr(int iChannel, int iNumber);
//float*** fTri_Arr(int iHeight, int iWidth, int iChannel);
//float**** fQuadri_Arr(int iHeight, int iWidth, int iChannel, int iNumber);
//
//void funcFree_Uni_Arr(float* fArr_Input);
//void funcFree_Du_Arr(int iChannel, float** fArr_Input);
//void funcFree_Tri_Arr(int iHeight, int iWidth, float*** fArr_Input);
//void funcFree_Quadri_Arr(int iHeight, int iWidth, int iChannel, float**** fArr_Input);
//
//int main(int argc, char* argv[]) {
//	// Allocate dynamic memories to each array
//	// Allocate the convolutional filter
//	fConv_0 = fQuadri_Arr(3, 3, 3, iChannel_01*iAlpha_Width); // 3 x 3 x 3 x 32
//
//															  // Allocate depth-wise convolutional filters
//	fConv_1_dw = fTri_Arr(3, 3, iChannel_01*iAlpha_Width); // 3 x 3 x 32 dw
//	fConv_2_dw = fTri_Arr(3, 3, iChannel_02*iAlpha_Width); // 3 x 3 x 64 dw
//	fConv_3_dw = fTri_Arr(3, 3, iChannel_03*iAlpha_Width); // 3 x 3 x 128 dw
//	fConv_4_dw = fTri_Arr(3, 3, iChannel_03*iAlpha_Width); // 3 x 3 x 128 dw
//	fConv_5_dw = fTri_Arr(3, 3, iChannel_04*iAlpha_Width); // 3 x 3 x 256 dw
//	fConv_6_dw = fTri_Arr(3, 3, iChannel_04*iAlpha_Width); // 3 x 3 x 256 dw
//														   // Iteration
//	fConv_7_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//	fConv_8_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//	fConv_9_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//	fConv_10_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//	fConv_11_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//
//	fConv_12_dw = fTri_Arr(3, 3, iChannel_05*iAlpha_Width); // 3 x 3 x 512 dw
//	fConv_13_dw = fTri_Arr(3, 3, iChannel_06*iAlpha_Width); // 3 x 3 x 1024 dw
//
//															// Allocate point-wise convolutional filters
//	fConv_1_pw = fDu_Arr(iChannel_01*iAlpha_Width, iChannel_02*iAlpha_Width); // 1 x 1 x 32 x 64 pw
//	fConv_2_pw = fDu_Arr(iChannel_02*iAlpha_Width, iChannel_03*iAlpha_Width); // 1 x 1 x 64 x 128 pw
//	fConv_3_pw = fDu_Arr(iChannel_03*iAlpha_Width, iChannel_03*iAlpha_Width); // 1 x 1 x 128 x 128 pw
//	fConv_4_pw = fDu_Arr(iChannel_03*iAlpha_Width, iChannel_04*iAlpha_Width); // 1 x 1 x 128 x 256 pw
//	fConv_5_pw = fDu_Arr(iChannel_04*iAlpha_Width, iChannel_04*iAlpha_Width); // 1 x 1 x 256 x 256 pw
//	fConv_6_pw = fDu_Arr(iChannel_04*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 256 x 512 pw
//																			  // Iteration
//	fConv_7_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
//	fConv_8_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
//	fConv_9_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
//	fConv_10_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
//	fConv_11_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_05*iAlpha_Width); // 1 x 1 x 512 x 512 pw
//
//	fConv_12_pw = fDu_Arr(iChannel_05*iAlpha_Width, iChannel_06*iAlpha_Width); // 1 x 1 x 512 x 1024 pw
//	fConv_13_pw = fDu_Arr(iChannel_06*iAlpha_Width, iChannel_06*iAlpha_Width); // 1 x 1 x 1024 x 1024 pw
//
//	fFully_0 = fDu_Arr(iChannel_06*iAlpha_Width, iResult_Class);
//
//	// Load trained weights from '.txt' files
//
//
//
//
//	while (1) {
//
//
//
//
//
//
//
//
//	}
//
//	// Memory release
//	//funcFreeArr(EXCEL_ROW, EXCEL_LINE, fConv_2_dw);
//
//	return 0;
//}
//
//
//
//
//
//
//float* fUni_Arr(int iChannel)
//{
//	int i;
//	float* ptr;
//
//	if ((ptr = (float*)malloc(iChannel * sizeof(float))) == NULL)
//	{
//		printf("\nMemory allocation failure\n");
//		exit(1);
//	}
//
//	// 모든 배열원소 초기화(중요하다)
//	for (i = 0; i < iChannel; i++) {
//		ptr[i] = 0;
//	}
//
//	printf("\nMEMORY ALLOCATION(char) OK!\n");
//	return ptr;
//}
//
//void funcFree_Uni_Arr(float* fArr_Input)
//{
//	free(fArr_Input);
//	return fArr_Input;
//}
//
//
//float** fDu_Arr(int iChannel, int iNumber)
//{
//	int i, j;
//	float** ptr;
//
//	if ((ptr = (float**)malloc(iChannel * sizeof(float*))) == NULL)
//	{
//		printf("\nMemory allocation failure\n");
//		exit(1);
//	}
//
//	for (i = 0; i < iChannel; i++) {
//		if ((ptr[i] = (float*)malloc(iNumber * sizeof(float))) == NULL)
//		{
//			printf("\nMemory allocation failure\n");
//			exit(1);
//		}
//	}
//
//	// 모든 배열원소 초기화(중요하다)
//	for (i = 0; i < iChannel; i++) {
//		for (j = 0; j < iNumber; j++) {
//			ptr[i][j] = 0;
//		}
//	}
//
//	printf("\nMEMORY ALLOCATION(char) OK!\n");
//	return ptr;
//}
//
//void funcFree_Du_Arr(int iChannel, float** fArr_Input)
//{
//	int i;
//	for (i = 0; i<iChannel; i++)
//	{
//		free(fArr_Input[i]);
//	}
//	free(fArr_Input);
//
//	return fArr_Input;
//}
//
//
//float*** fTri_Arr(int iHeight, int iWidth, int iChannel)
//{
//	int i, j, k;
//	float*** ptr;
//
//	if ((ptr = (float***)malloc(iHeight * sizeof(float**))) == NULL)
//	{
//		printf("\nMemory allocation failure\n");
//		exit(1);
//	}
//
//	for (i = 0; i < iHeight; i++) {
//		if ((ptr[i] = (float**)malloc(iWidth * sizeof(float*))) == NULL)
//		{
//			printf("\nMemory allocation failure\n");
//			exit(1);
//		}
//	}
//
//	for (i = 0; i < iHeight; i++) {
//		for (j = 0; j < iWidth; j++) {
//			if ((ptr[i][j] = (float*)malloc(iChannel * sizeof(float))) == NULL)
//			{
//				printf("\nMemory allocation failure\n");
//				exit(1);
//			}
//		}
//	}
//
//	// 모든 배열원소 초기화(중요하다)
//	for (i = 0; i < iHeight; i++) {
//		for (j = 0; j < iWidth; j++) {
//			for (k = 0; k < iChannel; k++) {
//				ptr[i][j][k] = 0;
//			}
//		}
//	}
//
//	printf("\nMEMORY ALLOCATION(char) OK!\n");
//	return ptr;
//}
//
//void funcFree_Tri_Arr(int iHeight, int iWidth, float*** fArr_Input)
//{
//	int i, j;
//	for (i = 0; i<iHeight; i++)
//	{
//		for (j = 0; j<iWidth; j++)
//		{
//			free(fArr_Input[i][j]);
//		}
//		free(fArr_Input[i]);
//	}
//	free(fArr_Input);
//
//	return fArr_Input;
//}
//
//
//float**** fQuadri_Arr(int iHeight, int iWidth, int iChannel, int iNumber)
//{
//	int i, j, k, l;
//	float**** ptr;
//
//	if ((ptr = (float****)malloc(iHeight * sizeof(float***))) == NULL)
//	{
//		printf("\nMemory allocation failure\n");
//		exit(1);
//	}
//
//	for (i = 0; i < iHeight; i++) {
//		if ((ptr[i] = (float***)malloc(iWidth * sizeof(float**))) == NULL)
//		{
//			printf("\nMemory allocation failure\n");
//			exit(1);
//		}
//	}
//
//	for (i = 0; i < iHeight; i++) {
//		for (j = 0; j < iWidth; j++) {
//			if ((ptr[i][j] = (float**)malloc(iChannel * sizeof(float*))) == NULL)
//			{
//				printf("\nMemory allocation failure\n");
//				exit(1);
//			}
//		}
//	}
//
//	for (i = 0; i < iHeight; i++) {
//		for (j = 0; j < iWidth; j++) {
//			for (k = 0; k < iChannel; k++) {
//				if ((ptr[i][j][k] = (float*)malloc(iNumber * sizeof(float))) == NULL)
//				{
//					printf("\nMemory allocation failure\n");
//					exit(1);
//				}
//			}
//		}
//	}
//
//	// 모든 배열원소 초기화(중요하다)
//	for (i = 0; i < iHeight; i++) {
//		for (j = 0; j < iWidth; j++) {
//			for (k = 0; k < iChannel; k++) {
//				for (l = 0; l < iNumber; l++) {
//					ptr[i][j][k][l] = 0;
//				}
//			}
//		}
//	}
//
//	printf("\nMEMORY ALLOCATION(char) OK!\n");
//	return ptr;
//}
//
//void funcFree_Quadri_Arr(int iHeight, int iWidth, int iChannel, float**** fArr_Input)
//{
//	int i, j, k;
//	for (i = 0; i<iHeight; i++)
//	{
//		for (j = 0; j<iWidth; j++)
//		{
//			for (k = 0; k < iChannel; k++) {
//				free(fArr_Input[i][j][k]);
//			}
//			free(fArr_Input[i][j]);
//		}
//		free(fArr_Input[i]);
//	}
//	free(fArr_Input);
//
//	return fArr_Input;
//}
//
//
//// Load File
//void funcLoad_Weight_3D(int iChannel, int iWidth, int iHeight, float*** fArr_Input, const char *cFilename[]) {
//
//	FILE *infile = fopen(cFilename, "rb");
//
//	if (!infile) {
//		printf("ERROR: File %s could not be opened! \n", cFilename);
//		exit(-1);
//	}
//
//	float* farrSerial_Information;
//	fread(farrSerial_Information, sizeof(float), iChannel*iWidth*iHeight, infile);
//	fclose(infile);
//
//	int iLayer_Temp = 0;
//	int iHeight_Temp = 0;
//	int iWidth_Temp = 0;
//	long int iIndex_Temp = 0;
//
//	for (iLayer_Temp = 0; iLayer_Temp < iChannel; iLayer_Temp++) {
//		for (iWidth_Temp = 0; iWidth_Temp < iWidth; iWidth_Temp++) {
//			for (iHeight_Temp = 0; iHeight_Temp < iHeight; iHeight_Temp++) {
//				fArr_Input[iLayer_Temp][iWidth_Temp][iHeight_Temp] = (float)farrSerial_Information[iIndex_Temp];
//				printf("%f\n", fArr_Input[iLayer_Temp][iWidth_Temp][iHeight_Temp]);
//				iIndex_Temp++;
//			}
//		}
//	}
//	return fArr_Input;
//}
//
