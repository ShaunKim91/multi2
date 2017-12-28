//#include "main.h"
//
//
//#define EXCEL_ROW 30100
//#define EXCEL_LINE 32
//#define CELL_DATASIZE 128
//
//char*** dataCube;
//
//int main() {
//
//	/* C에서 3차원 배열을 동적으로 할당 */
//	dataCube = TriArr(EXCEL_ROW, EXCEL_LINE, CELL_DATASIZE);
//	// 메모리 해제
//	FreeArr(EXCEL_ROW, EXCEL_LINE);
//
//	return 0;
//}
//
//char*** TriArr(int number, int height, int width)
//{
//	int i, j, k;
//	char*** ptr;
//
//	if ((ptr = (char***)malloc(number * sizeof(char**))) == NULL)
//	{
//		printf("\nMemory allocation failure\n");
//		exit(1);
//	}
//
//	for (i = 0; i < number; i++)
//		if ((ptr[i] = (char**)malloc(height * sizeof(char*))) == NULL)
//		{
//			printf("\nMemory allocation failure\n");
//			exit(1);
//		}
//
//	for (i = 0; i < number; i++)
//		for (j = 0; j < height; j++)
//			if ((ptr[i][j] = (char*)malloc(width * sizeof(char))) == NULL)
//			{
//				printf("\nMemory allocation failure\n");
//				exit(1);
//			}
//
//	// 모든 배열원소 초기화(중요하다)
//	for (i = 0; i < number; i++)
//		for (j = 0; j < height; j++)
//			for (k = 0; k < width; k++)
//				ptr[i][j][k] = '\0';
//
//	printf("\nMEMORY ALLOCATION(char) OK!\n");
//	return ptr;
//}
//
//
//void FreeArr(int number, int height)
//{
//	int i, j;
//	for (i = 0; i<number; i++)
//	{
//		for (j = 0; j<height; j++)
//		{
//			free(dataCube[i][j]);
//		}
//		free(dataCube[i]);
//	}
//	free(dataCube);
//}