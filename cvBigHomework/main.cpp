#include <iostream>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#define show
int main(int argc, char **argv) {
    if (argc !=3)
    {
	cout << "Please input the image path" << endl;
	return -1;
    }
    cout << "read src1 from " << argv[1] << endl;
    cout << "read src2 from " << argv[2] << endl<<endl;
    //read source image
    Mat src1,src2;
    src1 = imread( argv[1] );
    src2 = imread( argv[2] );
    if ( src1.empty() || src2.empty() )
    {
      cout<<"Failed to read image"<<endl;
      return -1;
    }
#ifdef show1
    imshow("src1",src1);
    imshow("src2",src2);
#endif
    Ptr<Feature2D> sift = xfeatures2d::SIFT::create(500,3,0.04,10,1.6);
    //预处理 
 //   resize(src1,src1,Size(),0.5,0.5);
 //   resize(src2,src2,Size(),0.5,0.5);
    GaussianBlur(src1,src1,Size(3,3),0);
    GaussianBlur(src2,src2,Size(3,3),0);
#ifdef show
    imshow("gaussian1",src1);
    imshow("gaussian2",src2);
#endif
    //提取特征点并计算描述子
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
    Mat descriptors1,descriptors2;
    sift->detectAndCompute(src1,Mat(),keypoints1,descriptors1);
    sift->detectAndCompute(src2,Mat(),keypoints2,descriptors2);
   // drawKeypoints(src1,keypoints1,src1,Scalar(0,0,255));
   // imshow("keypoints1",src1);
    //特征匹配
    FlannBasedMatcher matcher;
    vector<DMatch> matchPoints;
    matcher.match( descriptors1,descriptors2,matchPoints);
    //筛选匹配度高的点
    double min_dist = 10000, max_dist = -1;
    for ( int i = 0; i < descriptors1.rows; i++ )
    {
      double dist = matchPoints[i].distance;
      if ( dist < min_dist ) min_dist = dist;
      if ( dist > max_dist ) max_dist = dist;
    }
#ifdef show
     cout << " Max dist : " << max_dist << endl;
     cout << " Min dist : " << min_dist << endl<<endl;
#endif
     vector< DMatch > goodMatches;
     for ( int i = 0 ; i < descriptors1.rows; i++ )
     {
       if ( matchPoints[i].distance <= max ( 5 * min_dist, 30.0 ) ) //这里的系数根据匹配情况具体调整
	    goodMatches.push_back( matchPoints[i] );
    }
#ifdef show
      Mat img_match;
      drawMatches ( src1, keypoints1, src2, keypoints2, goodMatches, img_match );
      imshow ( "匹配到的特征点", img_match );
#endif 
      //计算本征矩阵
      cout<<" compute homo essential Mat ..." << endl << endl;
      Point2d principal_point ( 0,0);
      double focal_length = 4.0;
      vector<Point2f> points1,points2;
      for ( int i= 0; i < ( int )goodMatches.size(); i++ )
      {
	  points1.push_back( keypoints1[goodMatches[i].queryIdx].pt);
	  points2.push_back( keypoints2[goodMatches[i].trainIdx].pt);
      }
      Mat essential_mat,R,T;
      Mat homo;
      homo = findHomography( points1, points2, RANSAC );
      cout << " homo mat is : " << homo << endl<<endl;
  //    essential_mat = findEssentialMat ( points1, points2, focal_length, principal_point, RANSAC );
   //   cout<<" essential_matrix : " << essential_mat << endl;
   //   recoverPose ( essential_mat, points1,points2,R,T,focal_length,principal_point );
      
#ifdef show
  //    cout << " R is " << endl << R << endl;
  //    cout << " T is " << endl <<  T << endl;
#endif
      Mat result;
      
//      Point2f rPoint[4]={points1[0],points1[1],points1[2],points1[3]};
//      Point2f cPoint[4]={points2[0],points2[1],points2[2],points2[3]};
      cout<< " compute warpPerspective ... " << endl<<endl;
/*      Mat perMat = (Mat_<double>(3,3)<< 1.0, 0.0,  src1.cols, 
								0.0, 1.0, 0.0,
								0.0, 0.0, 1.0);
      perMat=perMat * homo;
      Mat rRect = ( Mat_<double>(4,3) <<0,0,1,
					0,1,1,
					1,0,1,
					1,1,1 );
     Point2f rPoint[4] = { Point2f(0,0), Point2f(0,1),
			Point2f(1,0), Point2f(1,1)
			};
      cout << "rRect" << endl <<  rRect.t() <<endl;
      Mat cRect;
      cRect=homo * rRect.t();
      cout << "cRect" << endl << cRect << endl;
      Point2f cPoint[4] = { Point2f(cRect.at<double>(0,0),cRect.at<double>(1,0)), Point2f(cRect.at<double>(0,1),cRect.at<double>(1,1)),
			Point2f(cRect.at<double>(0,2),cRect.at<double>(1,2)),Point2f(cRect.at<double>(0,3),cRect.at<double>(1,3)) 
		
			  };
			  */
			
//			cout << cPoint[0]<< endl
//			<<cPoint[1]<< endl
//			<<cPoint[2]<< endl
//			<<cPoint[3]<< endl;
	
//      Mat perMat;
//      perMat = getPerspectiveTransform(rPoint,cPoint);
	warpPerspective(src1,result,homo,Size(src2.cols+src1.cols, src2.rows+200));
//      warpPerspective(src1,rSrc1,essential_mat,src1.size()*4);
#ifdef show
      imshow (" mResult", result );
#endif 
  //    waitKey(0);
      cout << "Combine the image ... "<< endl << endl ;
      // find the edge
      Mat mw;
      int nonZerosNum =0;
      int location =0;
      Mat grayResult;
      cvtColor(result, grayResult, cv::COLOR_BGR2GRAY );
      Rect roi;
      for ( int i=1; i< src1.cols-2; i++ )
      {
	roi = Rect(Point(i,0),Point(i+1,src1.rows-1));
	mw = grayResult(roi);
	
	//rectangle ( result , roi, Scalar(0,0,255) );
	nonZerosNum = countNonZero ( mw );
	//   imshow (" mResult", result );
	// waitKey(0);
#ifdef showCount
	cout<< " countNonZero = " <<(double)nonZerosNum/(double)src1.rows << endl;
#endif	
	if (nonZerosNum > src1.rows*0.99)
	{
	  location = i+10;
#ifdef showLine
	  line(result,Point(i,0),Point(i,src1.rows),Scalar(0,0,255));
	   imshow (" mResult", result );
#endif
	  break;
	}
      }
      roi = Rect(Point(location,0),Point(src1.cols-1,src1.rows-1));
#ifdef showRect
      rectangle(result,roi,Scalar(0,0,255));
      imshow (" mResult", result );
#endif
      
      Point2f srcTri[] = {
	Point2f(0,0),
	Point2f(result.cols-1,0),
	Point2f(0,result.rows-1+100)
      };
      Point2f dstTri[] = {
	Point2f(0,5),
	Point2f(result.cols-1,5),
	Point2f(0,result.rows-1+105)
      };
      Mat warp_mat = getAffineTransform(srcTri,dstTri);
      Mat result1;
      warpAffine ( result,result1,warp_mat,result.size(),cv::INTER_LINEAR,BORDER_CONSTANT,Scalar());
#ifdef show
      imshow("result1",result1);
#endif
      
      Mat combImage=Mat::zeros(Size(result.cols,result.rows),CV_8UC3);
      Mat left,mid,right;
      left = combImage(Rect(Point(0,0),Point(location,src1.rows-1)));
      mid  = combImage(Rect(Point(location,0),Point(src1.cols,src1.rows-1)));
      right = combImage(Rect(Point(src1.cols,0),Point(combImage.cols-1,combImage.rows-1)));
   //   combImage(Rect(Point(0,0),Point(location,src1.rows-1)))=src2(Rect(Point(0,0),Point(location,src1.rows-1))).clone();

      
      src2(Rect(Point(0,0),Point(location,src1.rows-1))).copyTo(left);
      
      Mat combL,combR,comb;
      combL=src2(Rect(Point(location,0),Point(src1.cols,src1.rows-1))).clone();
      combR=result1(Rect(Point(location,0),Point(src1.cols,src1.rows-1))).clone();
      comb=Mat::zeros(combL.size(),CV_8UC3);
      double alpha,beta;
       for ( int i=0; i< combL.cols; i++ )
      {
	roi = Rect(Point(i,0),Point(i+1,src1.rows-1));
	beta=(double)i/((double)combL.cols);
	alpha = 1 - beta;
	cout<<"alpha: "<<alpha<<" beta: "<<beta<<endl;
	
	addWeighted(combL(roi),alpha,combR(roi),beta,0.0,comb(roi));
	//comb=
      }
#ifdef showComb
      imshow("combL",combL);
      imshow("combR",combR);
      imshow("comb",comb);
#endif
      comb.copyTo(mid);
      
      result1(Rect(Point(src1.cols,0),Point(combImage.cols-1,combImage.rows-1))).copyTo(right);
 //     src2(Rect(Point(location,0),Point(src1.cols-1,src1.rows-1))).copyTo(mid);
    
      imshow ("left" ,left);
      
 //     combImage(Rect(Point(0,0),Point(location,src1.rows-1)))=left;
      imshow ("combImage", combImage );
      while(waitKey(0)!=27);
      return 0;
}
