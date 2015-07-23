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
void DetectFourthLetter(Mat& image, Mat& extractedImage);
void ApplyTransformToAPoint(Point2f& point, Mat& transformMatrix);
void ExtractContoursFromColorImage(Mat& inputImage, Mat& outputImage,std::vector<std::vector<cv::Point>>& contours);
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

   // std::vector<std::vector<cv::Point>>::iterator itc = contours.begin();
    std::vector<cv::Point> poly;
    cv::Point2f originalBoxPoints[4];
    cv::Point2f topLeft = cv::Point2f(0,0);
    cv::Point2f topRight = cv::Point2f(0,0);
    cv::Point2f bottomRight = cv::Point2f(0,0);
    cv::Point2f bottomLeft = cv::Point2f(0,0);


    //cv::cvtColor(result, result, CV_GRAY2BGR);

    double angle = 0;
    ExtractBoxPolygon(result, contours, poly, angle, originalBoxPoints);
    //imshow("Function Transform", result);
    //waitKey(0);

/*    itc = contours.begin();
    while(itc != contours.end())
    {
      poly.clear();

      cv::approxPolyDP(*itc, poly, 10, true);

      if(poly.size() == 4)
      {

        double maxCosine = 0;
        for(int j=2; j<5; j++)
        {
          double cosine = fabs(FindAngle(poly[j%4], poly[j-2], poly[j-1]));
          maxCosine = MAX(maxCosine, cosine);
        }
        if(maxCosine<0.5)
        {

          polylines(result,poly,true,Scalar(255,255,255),5);
          originalBoxPoints[0] = poly[1];
          originalBoxPoints[1] = poly[0];
          originalBoxPoints[2] = poly[3];
          originalBoxPoints[3] = poly[2];
          FindVerticesOfBox(poly, topLeft, topRight, bottomRight, bottomLeft);
          ++itc;
          angle = FindAngle(bottomRight, cv::Point2f((bottomLeft.x+15), (bottomLeft.y)), bottomLeft);



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
    }*/

    /*for(int i=0; i<contours.size(); i++)
    {
      Scalar color = Scalar(rand()%256,rand()%256,rand()%256);

      cv::drawContours(result, contours,i, color,1);

    }*/



    //floodFill(result, Point(0,0), 0);

    //imshow("Flood Fill", result);

    Point2f referenceBoxPoints[4];
    GenerateReferenceBox(result, referenceBoxPoints);

    for(int corners=0; corners<4; corners++)
    {
      line(referenceBox, referenceBoxPoints[corners], referenceBoxPoints[(corners+1)%4], Scalar(0), 2, 8);
    }
    imshow("Reference Box", referenceBox);
    Point center = Point(result.cols/2, result.rows/2);
    cv::Mat center_rotation_matrix(2,3,CV_32FC1);
    center_rotation_matrix = getRotationMatrix2D(center,angle, 1.0);
   /* ApplyTransformToAPoint(originalBoxPoints[0], center_rotation_matrix);
    ApplyTransformToAPoint(originalBoxPoints[1], center_rotation_matrix);
    ApplyTransformToAPoint(originalBoxPoints[2], center_rotation_matrix);
    ApplyTransformToAPoint(originalBoxPoints[3], center_rotation_matrix);*/
    ApplyTransformToAPoint(topLeft, center_rotation_matrix);
    ApplyTransformToAPoint(topRight, center_rotation_matrix);
    ApplyTransformToAPoint(bottomRight, center_rotation_matrix);
    ApplyTransformToAPoint(bottomLeft, center_rotation_matrix);


  //  warpAffine(result, result, center_rotation_matrix, result.size());


    imshow("Center Rotation", result);
    waitKey(0);

    cv::Mat affine_matrix(3,3,CV_32FC1);
    //affine_matrix = getAffineTransform(originalBoxPoints, referenceBoxPoints);
    //warpAffine(result, result, affine_matrix, result.size());
    affine_matrix = getPerspectiveTransform(originalBoxPoints, referenceBoxPoints);
    //affine_matrix = getPerspectiveTransform(ExtractBoxPolygon(result, contours, poly, angle), referenceBoxPoints);
    warpPerspective(result, result, affine_matrix, result.size());

    imshow("Transformed", result);
    waitKey(0);

    cv::Mat extractedU(inputImage.size(),CV_8UC1,cv::Scalar(255));
    cv::Mat extractedUContours(inputImage.size(),CV_8UC1,cv::Scalar(255));


   /* DetectFourthLetter(result, extractedU);
    Point2f referenceUPoints[5];
    Point2f originalUPoints[5];

    referenceUPoints[0] = Point2f(180,233);
    referenceUPoints[1] = Point2f(200,275);
    referenceUPoints[2] = Point2f(211,231);
    referenceUPoints[3] = Point2f(215,281);
    referenceUPoints[4] = Point2f(184,283);
     for(int corners=0; corners<5; corners++)
     {
       line(extractedBox , referenceUPoints[corners], referenceUPoints[(corners+1)%5], Scalar(0), 1, 8);
     }

    std::vector<std::vector<cv::Point>> Ucontours;

    cv::findContours(extractedU,
                     Ucontours,
                     CV_RETR_LIST,
                     CV_CHAIN_APPROX_NONE);

    itc = Ucontours.begin();
    while(itc != Ucontours.end())
    {
      if(itc->size() < cmin || itc->size() > cmax)
      {
        itc = Ucontours.erase(itc);
      }
      else
      {
        ++itc;
      }

    }
    cv::drawContours(extractedUContours, Ucontours, -1, 128,2);
    cv::floodFill(extractedUContours, Point(0,0), 0);
    for(int i=0; i<extractedUContours.cols; i++)
    {
      for(int j=0; j<extractedUContours.rows; j++)
      {
        if(extractedUContours.at<uchar>(j,i) == 0)
        {
          extractedUContours.at<uchar>(j,i) = 255;
        }
        else
        {
          extractedUContours.at<uchar>(j,i) = 0;
        }
      }
    }
    Ucontours.clear();
    cv::findContours(extractedUContours,
                     Ucontours,
                     CV_RETR_LIST,
                     CV_CHAIN_APPROX_NONE);
    for(int i=0; i<extractedUContours.cols; i++)
    {
      for(int j=0; j<extractedUContours.rows; j++)
      {
        extractedUContours.at<uchar>(j,i) = 255;
      }
    }
    itc = Ucontours.begin();
    while(itc != Ucontours.end())
    {
      if(itc->size() < cmin || itc->size() > cmax)
      {
        itc = Ucontours.erase(itc);
      }
      else
      {
        ++itc;
      }

    }

    cv::drawContours(extractedUContours, Ucontours, -1, 0,2);
    cv::cvtColor(extractedUContours, extractedUContours, CV_GRAY2BGR);
    itc = Ucontours.begin();
    while(itc != Ucontours.end())
    {
      poly.clear();

      cv::approxPolyDP(*itc, poly, 10, true);
      printf("\rPoly size is %lu\n", poly.size());
      printf("\r0: %d %d\n",poly[0].x, poly[0].y);
      printf("\r1: %d %d\n",poly[1].x, poly[1].y);
      printf("\r2: %d %d\n",poly[2].x, poly[2].y);
      printf("\r3: %d %d\n",poly[3].x, poly[3].y);
      printf("\r4: %d %d\n",poly[4].x, poly[4].y);

      polylines(extractedUContours,poly,true,Scalar(0,0,255),1);
      if(poly.size() == 5)
      {
        originalUPoints[0] = poly[3];
        originalUPoints[1] = poly[4];
        originalUPoints[2] = poly[0];
        originalUPoints[3] = poly[1];
        originalUPoints[4] = poly[2];
      }
      ++itc;
    }
   // for(int corners=0; corners<5; corners++)
   // {
   //   line(extractedUContours, referenceUPoints[corners], referenceUPoints[(corners+1)%5], Scalar(0), 1, 8);
   // }

    imshow("Extracted U",extractedUContours);
    cv::Mat new_affine_matrix(3,3,CV_32FC1);

    new_affine_matrix = getAffineTransform(originalUPoints, referenceUPoints);
    warpAffine(extractedUContours, extractedUContours, new_affine_matrix, extractedUContours.size());
    //warpAffine(result, result, new_affine_matrix, result.size());
    imshow("New Transformed",result);
    imshow("Transformed U", extractedUContours);
*/
    FillInHollowLetters(result);

    imshow("Hollow Letters", result);


    for(int i=0; i<result.cols; i++)
    {
      for(int j=0; j<result.rows; j++)
      {
        if(result.at<uchar>(j,i) == 0)
        {
          result.at<uchar>(j,i) = 255;
        }
        else
        {
          result.at<uchar>(j,i) = 0;
        }
      }
    }

    imshow("Filled contours", result);

    SweepForNoise(result);
    imshow("Noise Reduced Contours", result);


    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    tess.SetImage((uchar*)result.data, result.cols, result.rows, 1, result.cols);
        
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
        std::cout<<x<<" "<<y<<" "<<w<<" "<<h<<std::endl;
        cv::rectangle(result, Point(x,y), Point((x+w),(y+h)), Scalar(0,255,255),1,8,0);
      }
    }
    else
    {
      printf("\rNo text detected\n");
    }
    imshow("OCR'd Image", result);

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

  cv::Point2f topLeft;
  cv::Point2f topRight;
  cv::Point2f bottomRight;
  cv::Point2f bottomLeft;
  angle = 0;
  while(itc != contours.end())
  {
    poly.clear();

    cv::approxPolyDP(*itc, poly, 10, true);

    if(poly.size() == 4)
    {

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
        angle = FindAngle(bottomRight, cv::Point2f((bottomLeft.x+15), (bottomLeft.y)), bottomLeft);
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
  imshow("Function Thing?", inputImage);

  Point center = Point(inputImage.cols/2, inputImage.rows/2);
  cv::Mat center_rotation_matrix(2,3,CV_32FC1);
  center_rotation_matrix = getRotationMatrix2D(center,(-1*angle), 1.0);
  ApplyTransformToAPoint(topLeft, center_rotation_matrix);
  ApplyTransformToAPoint(topRight, center_rotation_matrix);
  ApplyTransformToAPoint(bottomRight, center_rotation_matrix);
  ApplyTransformToAPoint(bottomLeft, center_rotation_matrix);
 // warpAffine(inputImage, inputImage, center_rotation_matrix, inputImage.size());

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

void DetectFourthLetter(Mat& image, Mat& extractedImage)
{
  bool letterToggle = false;
  bool hitTheFirstLetter = false;
  bool hitTheSecondLetter = false;
  bool hitTheThirdLetter = false;
  bool hitTheFourthLetter = false;
  for(int i=0; i<image.cols; i++)
  {
    int whiteCount = 0;

    for(int j=0; j<image.rows; j++)
    {
      if((image.at<uchar>(j,i) == 255) || (image.at<uchar>(j,(i-1)) == 255) || (image.at<uchar>(j,(i+1)) == 255))
      {
        whiteCount++;
      }
      if((image.at<uchar>(j,i) == 255) && !hitTheFirstLetter)
      {
        letterToggle = true;
        hitTheFirstLetter = true;
      }
      if((image.at<uchar>(j,i) == 255) && hitTheFirstLetter && !letterToggle && !hitTheSecondLetter)
      {
        letterToggle = true;
        hitTheSecondLetter = true;
      }
      if((image.at<uchar>(j,i) == 255) && hitTheSecondLetter && !letterToggle && !hitTheThirdLetter)
      {
        letterToggle = true;
        hitTheThirdLetter = true;
      }
      if((image.at<uchar>(j,i) == 255) && hitTheThirdLetter && !letterToggle && !hitTheFourthLetter)
      {
        letterToggle = true;
        hitTheFourthLetter = true;
      }
      if((image.at<uchar>(j,i) == 255) && hitTheFourthLetter && letterToggle)
      {
        extractedImage.at<uchar>(j,i) = 0;
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

