// ProcessingTests
// This program exists to test image processing techniques and run them
// through a Haar cascade object detector.

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <tesseract/strngs.h>

#define _USE_MATH_DEFINES

using namespace cv;

//Function prototypes
void Usage(void);
void SaveFile(Mat& image, const char* path, const string& description);
double FindAngle( cv::Point pt1, cv::Point pt2, cv::Point pt0 );
void FillInHollowLetters(Mat& image);
void SweepForNoise(Mat& image);
void SegmentImage(Mat& image,
                  Mat& firstLetter,
                  Mat& secondLetter,
                  Mat& thirdLetter,
                  Mat& fourthLetter,
                  Mat& segmentedImage);

void ApplyTransformToAPoint(Point2f& point, Mat& transformMatrix);
void ExtractContoursFromColorImage(Mat& inputImage,
                                   Mat& outputImage,
                                   std::vector<std::vector<cv::Point>>& contours);

void ExtractBoxPolygon(Mat& inputImage,
                       std::vector<std::vector<cv::Point>>& contours,
                       std::vector<cv::Point>& poly,
                       double& angle,
                       cv::Point2f* originalBoxPoints);

void FindVerticesOfBox(std::vector<cv::Point> poly,
                       cv::Point2f& topLeft,
                       cv::Point2f& topRight,
                       cv::Point2f& bottomRight,
                       cv::Point2f& bottomLeft);

void GenerateReferenceBox(Mat& inputImage, cv::Point2f* referenceBoxPoints);
void TesseractOCR(Mat& inputImage);
void GenerateRotationMatrix(Mat& inputImage,
                            double& angle,
                            cv::Point2f* topLeft,
                            cv::Point2f* topRight,
                            cv::Point2f* bottomRight,
                            cv::Point2f* bottomLeft);
void CorrectSmearing(Mat& inputImage);
void CleanIndividualLetter(Mat& letterImage);
void CleanSegmentedImage(Mat& segmentedImage);

int main(int argc, char * const * argv)
{
    
  int ch = 0;
    
  char *xml_filename = nullptr;
    
  while ((ch = getopt(argc, argv, "dx:")) != -1)
  {
    switch (ch) {
      case 'x':
        xml_filename = optarg;
        break;
      default:
        Usage();
    }
  }
  argc -= optind;
  argv += optind;
    
  if (!xml_filename || argc == 0)
  {
    Usage();
  }
    
  //Make empty image structures
  cv::Mat inputImage;

  //Enumerate through the image arguments. argv[1] is the detector xml file
  for(int input=0; input<(argc); input++)
  {
        
    inputImage = cv::imread(argv[input]);    //Read the image

    if(inputImage.empty())                   //Prevent running on an empty image
    {
      return true;
    }

    imshow("Original Image", inputImage);

    std::vector<std::vector<cv::Point>> contours;
    cv::Mat result(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat extractedBox(inputImage.size(), CV_32FC1, cv::Scalar(255));
    cv::Mat referenceBox(inputImage.size(), CV_32FC1, cv::Scalar(255));

    ExtractContoursFromColorImage(inputImage, result, contours);

    std::vector<cv::Point> poly;
    cv::Point2f originalBoxPoints[4];
    double angle = 0;

    ExtractBoxPolygon(result, contours, poly, angle, originalBoxPoints);

    Point2f referenceBoxPoints[4];
    GenerateReferenceBox(result, referenceBoxPoints);


    cv::Mat affine_matrix(3,3,CV_32FC1);
    affine_matrix = getPerspectiveTransform(originalBoxPoints, referenceBoxPoints);
    warpPerspective(result, result, affine_matrix, result.size());


    for(int j=0; j<result.rows; j++)
    {
      for(int i=0; i<result.cols; i++)
      {
        if(result.at<uchar>(j,i) != 0)
        {
          if(result.at<uchar>(j,i) >= 10 && result.at<uchar>(j,i) <= 150)
          {
            result.at<uchar>(j,i) = 128;
          }
          else if(result.at<uchar>(j,i) > 150)
          {
            result.at<uchar>(j,i) = 255;
          }
        }
      }
    }
    FillInHollowLetters(result);
    CorrectSmearing(result);

    cv::Mat firstLetter(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat secondLetter(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat thirdLetter(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat fourthLetter(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat segmentedImage(inputImage.size(),CV_8UC1,cv::Scalar(255));

    SegmentImage(result, firstLetter, secondLetter, thirdLetter, fourthLetter, segmentedImage);
    CleanIndividualLetter(firstLetter);
    CleanIndividualLetter(secondLetter);
    CleanIndividualLetter(thirdLetter);
    CleanIndividualLetter(fourthLetter);
    CleanSegmentedImage(segmentedImage);

    TesseractOCR(firstLetter);
    TesseractOCR(secondLetter);
    TesseractOCR(thirdLetter);
    TesseractOCR(fourthLetter);
    TesseractOCR(segmentedImage);


  }
  waitKey(0);
    
  return 0;
}

//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//----------------------------------FUNCTIONS-----------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------
// This function takes a segmented image and processes it for OCR
void CleanSegmentedImage(Mat& segmentedImage)
{
  floodFill(segmentedImage, Point(0,0), 0);
  floodFill(segmentedImage, Point(719,0), 0);
  for(int i=0; i<segmentedImage.cols; i++)
  {
    for(int j=0; j<segmentedImage.rows; j++)
    {
      if(segmentedImage.at<uchar>(j,i) == 0)
      {
        segmentedImage.at<uchar>(j,i) = 255;
      }
      else
      {
        segmentedImage.at<uchar>(j,i) = 0;
      }
    }
  }
  SweepForNoise(segmentedImage);
}

//------------------------------------------------------------------------------------
// This function takes an image of only one letter and processes it for OCR
void CleanIndividualLetter(Mat& letterImage)
{
  floodFill(letterImage, Point(0,0), 0);
  floodFill(letterImage, Point(719,0), 0);
  for(int i=0; i<letterImage.cols; i++)
  {
    for(int j=0; j<letterImage.rows; j++)
    {
      if(letterImage.at<uchar>(j,i) == 0)
      {
        letterImage.at<uchar>(j,i) = 255;
      }
      else
      {
        letterImage.at<uchar>(j,i) = 0;
      }
    }
  }
}


//------------------------------------------------------------------------------------
// This function should correct the smearing inside of characters
void CorrectSmearing(Mat& inputImage)
{
  for(int i=0; i<inputImage.cols;i++)
  {
    for(int j=1; j<inputImage.rows-1;j++)
    {
      if( (inputImage.at<uchar>(j,i) == 0) &&
        ((inputImage.at<uchar>((j+1),i) == 255) ||
        (inputImage.at<uchar>((j-1),i) == 255)) )
      {
        inputImage.at<uchar>(j,i) = 255;
      }
    }
  }

}


//------------------------------------------------------------------------------------
// This function generates the matrix necessary for rotating the image.
// TODO: Make this function actually work
void GenerateRotationMatrix(Mat& inputImage,
                            double& angle,
                            cv::Point2f* topLeft,
                            cv::Point2f* topRight,
                            cv::Point2f* bottomRight,
                            cv::Point2f* bottomLeft)
{
  // This allows me to rotate about the center of the image rather than the origin
  Point center = Point(inputImage.cols/2, inputImage.rows/2);
  cv::Mat center_rotation_matrix(2,3,CV_32FC1);
  center_rotation_matrix = getRotationMatrix2D(center,((-1*(angle*(180/M_PI)/6))), 0.8);
  ApplyTransformToAPoint(*topLeft, center_rotation_matrix);
  ApplyTransformToAPoint(*topRight, center_rotation_matrix);
  ApplyTransformToAPoint(*bottomRight, center_rotation_matrix);
  ApplyTransformToAPoint(*bottomLeft, center_rotation_matrix);
  warpAffine(inputImage, inputImage, center_rotation_matrix, inputImage.size());
}

//------------------------------------------------------------------------------------
// This function runs the OCR package on an image and prints the text detected.
void TesseractOCR(Mat& inputImage)
{
  tesseract::TessBaseAPI tess;
  tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
  tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
  tess.SetImage((uchar*)inputImage.data, inputImage.cols, inputImage.rows, 1, inputImage.cols);

  // Get the text
  char* out = tess.GetUTF8Text();
  std::cout << out << std::endl;
  //box
  Boxa* bounds = tess.GetWords(NULL);
  if(bounds)
  {
    l_int32 count = bounds->n;
    for(int i=0; i<count; i++)
    {
      Box* b = bounds->box[i];
      int x = b->x;
      int y = b->y;
      int w = b->w;
      int h = b->h;
      //std::cout<<x<<" "<<y<<" "<<w<<" "<<h<<std::endl;
      cv::rectangle(inputImage, Point(x,y), Point((x+w),(y+h)), Scalar(0,255,255),1,8,0);
    }
  }
  else
  {
    printf("\rNo text detected\n");
  }
  imshow("OCR'd Image", inputImage);

}

//------------------------------------------------------------------------------------
// This function generates a reference box to use in transformations
void GenerateReferenceBox(Mat& inputImage, cv::Point2f* referenceBoxPoints)
{
  referenceBoxPoints[0] = cv::Point2f((inputImage.cols*14/15),(inputImage.rows*7/16));
  referenceBoxPoints[1] = cv::Point2f((inputImage.cols*15/15),(inputImage.rows*7/16));
  referenceBoxPoints[2] = cv::Point2f((inputImage.cols*15/15),(inputImage.rows*9/16));
  referenceBoxPoints[3] = cv::Point2f((inputImage.cols*14/15),(inputImage.rows*9/16));
}





//------------------------------------------------------------------------------------
// This function scans the contours of an image and looks for the warped box.
// TODO: Expand this to search for any polygon instead of only a box.
void ExtractBoxPolygon(Mat& inputImage,
                       std::vector<std::vector<cv::Point>>& contours,
                       std::vector<cv::Point>& poly,
                       double& angle,
                       cv::Point2f* originalBoxPoints)
{
  std::vector<std::vector<cv::Point>>::iterator itc = contours.begin();
  Point center = Point(inputImage.cols/2, inputImage.rows/2);
  cv::Point2f topLeft;
  cv::Point2f topRight;
  cv::Point2f bottomRight;
  cv::Point2f bottomLeft;
  angle = 0;
  while(itc != contours.end())
  {
    poly.clear();

    cv::approxPolyDP(*itc, poly, 10, true); // Draws polygons around each contour

    if(poly.size() == 4)
    {
      // Determine if this polygon is the box
      double maxCosine = 0;
      for(int j=2; j<5; j++)
      {
        double cosine = fabs(FindAngle(poly[j%4], poly[j-2], poly[j-1]));
        maxCosine = MAX(maxCosine, cosine);
      }
      if(maxCosine<0.5)
      {

        polylines(inputImage,poly,true,Scalar(255,255,255),5);
        originalBoxPoints[0] = poly[1];
        originalBoxPoints[1] = poly[0];
        originalBoxPoints[2] = poly[3];
        originalBoxPoints[3] = poly[2];
        FindVerticesOfBox(poly, topLeft, topRight, bottomRight, bottomLeft);
        //angle = FindAngle(bottomRight, cv::Point2f((bottomLeft.x+15), (bottomLeft.y)), bottomLeft);
        angle = acos(FindAngle(bottomRight, cv::Point2f((center.x+15), (center.y)), center));
        printf("\r Calculated angle: %f radians\n", angle);
        printf("\r Which is: %f degrees\n", (angle*(180/M_PI)));
        ++itc;
      }
      else
      {
        ++itc;
      }
    }
    else
    {
      ++itc;
    }
  }
  floodFill(inputImage, Point(0,0), 0);

  GenerateRotationMatrix(inputImage, angle, &topLeft, &topRight, &bottomRight, &bottomLeft);

}


//------------------------------------------------------------------------------------
// This function takes a BGR color image and extracts the contours from it.
void ExtractContoursFromColorImage(Mat& inputImage, Mat& outputImage,std::vector<std::vector<cv::Point>>& contours)
{
  cv::cvtColor(inputImage, inputImage, CV_BGR2GRAY);
  Canny(inputImage, inputImage, 100, 200);
  cv::findContours(inputImage,
                   contours,
                   CV_RETR_LIST,
                   CV_CHAIN_APPROX_NONE);

  int cmin = 50;
  int cmax = 1000;
  std::vector<std::vector<cv::Point>>::iterator itc = contours.begin();
  while(itc != contours.end())
  {
    if(itc->size() < cmin || itc->size() > cmax)
    {
      itc = contours.erase(itc);
    }
    else
    {
      ++itc;
    }
  }
  cv::drawContours(outputImage, contours, -1, 128,2);
}

//------------------------------------------------------------------------------------
// This function determines which vertices of a box correspond to which location
// in plain terms.
// TODO: Dynamically determine which point is which
void FindVerticesOfBox(std::vector<cv::Point> poly,
                       cv::Point2f& topLeft,
                       cv::Point2f& topRight,
                       cv::Point2f& bottomRight,
                       cv::Point2f& bottomLeft)
{
  topLeft = poly[1];
  topRight = poly[0];
  bottomRight = poly[3];
  bottomLeft = poly[2];
}
//------------------------------------------------------------------------------------
// This function will apply a matrix transformation to individual points
void ApplyTransformToAPoint(Point2f& point, Mat& transformMatrix)
{
  Mat_<double> pointMatrix(3,1);
  Mat_<double> pointResult(3,1);
  pointMatrix << point.x,point.y,1.0;
  pointResult = transformMatrix * pointMatrix;
  point.x = pointResult(0);
  point.y = pointResult(1);
}


//------------------------------------------------------------------------------------
// This function will detect the fourth letter in the image.

void SegmentImage(Mat& inputImage,
                  Mat& firstLetter,
                  Mat& secondLetter,
                  Mat& thirdLetter,
                  Mat& fourthLetter,
                  Mat& segmentedImage)
{
  bool letterToggle = false;
  bool hitTheFirstLetter = false;
  bool hitTheSecondLetter = false;
  bool hitTheThirdLetter = false;
  bool hitTheFourthLetter = false;
  bool beyondTheFourthLetter = false;

  for(int i=0; i<inputImage.cols; i++)
  {
    int whiteCount = 0;

    for(int j=0; j<inputImage.rows; j++)
    {
      if((inputImage.at<uchar>(j,i) == 255) || (inputImage.at<uchar>(j,(i-1)) == 255) || (inputImage.at<uchar>(j,(i+1)) == 255))
      {
        whiteCount++;
      }
      if((inputImage.at<uchar>(j,i) == 255) && !hitTheFirstLetter)
      {
        letterToggle = true;
        hitTheFirstLetter = true;
      }
      if(hitTheFirstLetter && !hitTheSecondLetter)
      {
        firstLetter.at<uchar>(j,i) = inputImage.at<uchar>(j,(i-5));
      }
      if((inputImage.at<uchar>(j,i) == 255) && hitTheFirstLetter && !letterToggle && !hitTheSecondLetter)
      {
        letterToggle = true;
        hitTheSecondLetter = true;
      }
      if(hitTheSecondLetter && !hitTheThirdLetter)
      {
        secondLetter.at<uchar>(j,i) = inputImage.at<uchar>(j,(i-5));
      }
      if((inputImage.at<uchar>(j,i) == 255) && hitTheSecondLetter && !letterToggle && !hitTheThirdLetter)
      {
        letterToggle = true;
        hitTheThirdLetter = true;
      }
      if(hitTheThirdLetter && !hitTheFourthLetter)
      {
        thirdLetter.at<uchar>(j,i) = inputImage.at<uchar>(j,(i-5));
      }
      if((inputImage.at<uchar>(j,i) == 255) && hitTheThirdLetter && !letterToggle && !hitTheFourthLetter)
      {
        letterToggle = true;
        hitTheFourthLetter = true;
      }
      if(hitTheFourthLetter && letterToggle)
      {
        fourthLetter.at<uchar>(j,i) = inputImage.at<uchar>(j,(i-1));
      }
      if(hitTheFourthLetter && !letterToggle)
      {
        beyondTheFourthLetter = true;
      }
      if(hitTheFourthLetter && beyondTheFourthLetter)
      {
        segmentedImage.at<uchar>(j,i) = inputImage.at<uchar>(j,i);
      }
    }
    if(whiteCount == 0)
    {
      letterToggle = false;
    }
  }
}

//------------------------------------------------------------------------------------
// This function will color in hollow letters - O, A, etc.

void FillInHollowLetters(Mat& image)
{
  for(int j=0; j<image.rows; j++)
  {
    bool inALetter = false;
    bool inAHollowLetter = false;
    bool hitWhiteSpace = false;
    bool fillingInHasBegun = false;
    for(int i=0; i<(image.cols-1); i++)
    {

      if(image.at<uchar>(j,(i)) == 0)
      {
        inALetter = false;
        inAHollowLetter = false;
        hitWhiteSpace = false;
        fillingInHasBegun = false;
      }
      if((image.at<uchar>(j,i) == 255) && inALetter)
      {
        hitWhiteSpace = true;
      }
      if((image.at<uchar>(j,i) == 128) && hitWhiteSpace && inALetter)
      {
        inAHollowLetter = true;
      }
      if((image.at<uchar>(j,i) == 255) && inAHollowLetter)
      {
        image.at<uchar>(j,i) = 0;
        fillingInHasBegun = true;
      }
      if((image.at<uchar>(j,i) == 128) && !inALetter)
      {
        inALetter = true;
      }
      if((image.at<uchar>(j,i) == 128) && fillingInHasBegun)
      {
        inAHollowLetter = false;
        hitWhiteSpace = false;
      }

    }
  }
}

//------------------------------------------------------------------------------------
// This function sweeps for noise by erasing rows and columns that
// contain no letters.

void SweepForNoise(Mat& image)
{
  int blackPixelCount = 0;
  for(int j=0; j<image.rows; j++)
  {
    blackPixelCount = 0;
    for(int i=0; i<image.cols; i++)
    {
      if(image.at<uchar>(j,i) == 0)
      {
        blackPixelCount++;
      }
    }
    if(blackPixelCount < 10)
    {
      for(int i=0; i<image.cols; i++)
      {
        image.at<uchar>(j,i) = 255;
      }
    }
  }

  for(int i=0; i<image.cols; i++)
  {
    blackPixelCount = 0;
    for(int j=0; j<image.rows; j++)
    {
      if(image.at<uchar>(j,i) == 0)
      {
        blackPixelCount++;
      }
    }
    if(blackPixelCount < 10)
    {
      for(int j=0; j<image.rows; j++)
      {
        image.at<uchar>(j,i) = 255;
      }
    }
  }
}

//------------------------------------------------------------------------------------
// This function will save the images as new files.
// This is only useful for analysis.

void SaveFile(Mat& image, const char* path, const string& description)
{
  char newFileName[1024];
  snprintf(newFileName, 1024, "%s.%s.jpg", path, description.c_str());
  cv::imwrite(newFileName, image);
}

//--------------------------------------------------------------------------------------
// This is a thing for getopt. It gives instructions for how to use the program.

void Usage()
{
  printf("Processing Tests -x <training file> <filename> [...]\n");
  printf("   -x xml training file\n");
  exit(-1);
}

//--------------------------------------------------------------------------------------
// This function calculates the cosine of the angle between two lines based on their points.
double FindAngle( cv::Point pt1, cv::Point pt2, cv::Point pt0 )
{
  double dx1 = pt1.x - pt0.x;
  double dy1 = pt1.y - pt0.y;
  double dx2 = pt2.x - pt0.x;
  double dy2 = pt2.y - pt0.y;
  return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2));
}

