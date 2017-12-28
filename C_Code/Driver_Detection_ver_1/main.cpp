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
//	/* C���� 3���� �迭�� �������� �Ҵ� */
//	dataCube = TriArr(EXCEL_ROW, EXCEL_LINE, CELL_DATASIZE);
//	// �޸� ����
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
//	// ��� �迭���� �ʱ�ȭ(�߿��ϴ�)
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