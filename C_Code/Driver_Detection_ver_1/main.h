//#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include<conio.h> // For exiting program by using ESC button
#include <windows.h> // Load images (only for Windows OS)
#include <time.h>  // Fot time-stamp     

// Load images (only for Windows OS)
BITMAPFILEHEADER	Header_File;
BITMAPINFOHEADER	Header_Info;
// Size of RGB images (only for test)
#define	X_Size	224
#define	Y_Size	224

// Input width
#define iAlpha_Width 0.5

// Number of result classes
#define iResult_Class 2 // Binary result

// Input image resolution
#define iResolution_ininital 224
#define iResolution_01 112
#define iResolution_02 56
#define iResolution_03 28
#define iResolution_04 14
#define iResolution_05 7

// Input image channel size
#define iChannel_01 32
#define iChannel_02 64
#define iChannel_03 128
#define iChannel_04 256
#define iChannel_05 512
#define iChannel_06 1024

// Batch normalization
#define fGamma_Batch 0
#define fBeta_Batch 0

// Number of convolutional layers
#define iConv_General_NUM 1
#define iConv_Seperable_NUM 13
#define iFully_Connected_NUM 1