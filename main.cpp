#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "func.cpp"

//opencv함수를 c++로 구현한 코드


int main() 
{

    printf("hello");
   //소벨, 캐니, 모폴로지, 라벨링
   //카메라인식, 딥러닝, 트래킹

    // 1. grayscale	    2. bgr2hsv	    3. average blurring 	4. gausian blurring	    5. right rotation 	
    // 6. sharpening	    7. reset	8. left rotation	    9. Thresholding    10. Thresholding invert version
    // 11. sobel        12. canny       13. erosion             14. delitation      15. cca
    


    cv::Mat image_g = cv::imread("lena.jpeg", cv::IMREAD_GRAYSCALE);
	cv::Mat image_c = cv::imread("lena.jpeg", cv::IMREAD_COLOR);

	
	int width = image_g.cols;
    int height = image_g.rows;
    int size = width * height;

    int sizeC = width * height * 3;	    //color이미지를 위한 size
	
   
	
    
	int* image_color = new int[sizeC];      //Mat 형식의 color이미지 행렬로 복사
    for (int i = 0; i < size; i++) 
    {
        
			cv::Vec3b& pixel = image_c.at<cv::Vec3b>(i / width, i % width);
			
            image_color[i] = pixel[0];
			image_color[size + i] = pixel[1];
			image_color[size+ size+ i] = pixel[2];
            
        
    }

	
	

// 1. grayscale

    int* image_gray = new int[size];    

	gogray(image_color,image_gray, width, height);  

    cv::Mat img_cal_gray(height, width, CV_8UC1);       //grayscale된 행렬을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_gray.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_gray[i]);
        
    }
    cv::imshow("img_cal_gray",img_cal_gray);

    


//2. bgr2hsv    

    double* image_hsv = new double[sizeC];
    

    bgr_hsv(image_color,image_hsv, width, height);
    

    cv::Mat img_cal_hsv(height, width, CV_8UC3);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {                
        
        img_cal_hsv.at<cv::Vec3b>((i/width), (i%width))[0] = static_cast<uchar>(image_hsv[i]);
        img_cal_hsv.at<cv::Vec3b>((i/width), (i%width))[1] = static_cast<uchar>(image_hsv[i+size]);
        img_cal_hsv.at<cv::Vec3b>((i/width), (i%width))[2] = static_cast<uchar>(image_hsv[i+size+size]);
            
        
    }

    	
//3. average blurring    
	
    int kernelSize = 5;


    double* image_b_aver= new double[size];		    //Mat에서 받은 이미지의 각 픽셀값을 2차원 포인터 배열에 복사
    for (int i = 0; i < size; i++) 
    {			
        
        image_b_aver[i] = image_g.at<uchar>(i / width, i % width);
        
    }
    
    // 이미지 블러링 수행
    blur_aver(image_b_aver, width, height,kernelSize);

    // 결과 이미지 출력
    cv::Mat img_cal_blur_aver(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_blur_aver.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_b_aver[i]);
        
    }


//4. gausian blurring




    double* image_b_gausian= new double[size];		//Mat에서 받은 이미지의 각 픽셀값을 2차원 포인터 배열에 복사
    for (int i = 0; i < size; i++) 
    {			
        
        image_b_gausian[i] = image_g.at<uchar>(i / width, i % width);
        
    }

    int ksize = 5;

    blur_gausian(image_b_gausian, width, height);


    cv::Mat img_cal_blur_gausian(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_blur_gausian.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_b_gausian[i]);
        
    }

    cv::imshow("gausian", img_cal_blur_gausian);


//5. sharpening

    sharpenImage(image_b_gausian,width,height);
    cv::Mat img_cal_blur_sharp(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_blur_sharp.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_b_gausian[i]);
        
    }

    cv::imshow("sharp", img_cal_blur_sharp);






//6. right rotation
    

    int* img1 = new int[size];
    for (int i = 0; i < size; i++) 
    {
        img1[i] = image_g.at<uchar>(i / width, i % width);
    }

    rotateImage_90R(img1, width, height);

    cv::Mat img_cal_90R(height, width, CV_8UC1);
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_90R.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img1[i]);
        
    }

    cv::imshow("Rotated Image_R", img_cal_90R);



//7. reset


    int* img_reset = new int[size];		
    for (int i = 0; i < size; i++) 
    {
        
        img_reset[i] = image_g.at<uchar>(i / width, i % width);
        
    }
    reset(img1,img_reset,width, height );



//8. left rotation
	

	rotateImage_90L(img1, width, height);

    cv::Mat img_cal_90L(height, width, CV_8UC1);
    for (int i = 0; i < size; i++)
    {
        
        img_cal_90L.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img1[i]);
        
    }

    
    cv::imshow("Rotated Image_L", img_cal_90L);
    

    

    reset(img1,img_reset,width, height );
	

	


//9. Thresholding    
    int threshold = 127;
	Thresh(img1, width, height, threshold);


	cv::Mat img_cal_thr(height, width, CV_8UC1);
    for (int i = 0; i < size; i++)
    {
        
        img_cal_thr.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img1[i]);
        
    }

    cv::imshow("cal_thr", img_cal_thr);




	reset(img1,img_reset,width, height );
	


//10. Thresholding invert version	


	Thres_IVT(img1, width, height, threshold);

	cv::Mat img_cal_thr_IVT(height, width, CV_8UC1);
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_thr_IVT.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img1[i]);
        
    }
    cv::imshow("cal_thr_IVT", img_cal_thr_IVT);
    reset(img1,img_reset,width, height );


	
    

//11. sobel


    double* img_sobel = new double[size];		
    for (int i = 0; i < size; i++) 
    {
        
        img_sobel[i] = image_g.at<uchar>(i / width, i % width);
        
    }


    sobel(img_sobel, width, height);

    cv::Mat img_cal_sobel(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_sobel.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img_sobel[i]);
        
    }





//12. canny

    double* img_cann = new double[size];
    for (int i = 0; i < size; i++) 
    {
        
        img_cann[i] = image_g.at<uchar>(i / width, i % width);
        
    }

    canny(img_cann, width, height);

    

    cv::Mat img_cal_canny(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        img_cal_canny.at<uchar>((i/width), (i%width)) = static_cast<uchar>(img_cann[i]);
        
    }



//13. erosion

    cv::Mat image_morphology = cv::imread("lena.jpeg",cv::IMREAD_GRAYSCALE);
    if (image_morphology.empty()) 
    {
        std::cout << "Failed to load image." << std::endl;
        return -1;
    }
    


    int* image_e= new int[size];		    //Mat에서 받은 이미지의 각 픽셀값을 2차원 포인터 배열에 복사
    for (int i = 0; i < size; i++) 
    {			
        
        image_e[i] = image_morphology.at<uchar>(i / width, i % width);
        
    }

    Thresh(image_e, width, height, 127);
    
    
    erosion(image_e,width, height);

    cv::Mat er(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        er.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_e[i]);
        
    }


    cv::imshow("erosion", er);


//14. delitation

    int* image_d= new int[size];		    //Mat에서 받은 이미지의 각 픽셀값을 2차원 포인터 배열에 복사
    for (int i = 0; i < size; i++) 
    {			
        
        image_d[i] = image_morphology.at<uchar>(i / width, i % width);
        
    }

    Thresh(image_d, width, height, 127);

    delitation(image_d, width, height);
    

    cv::Mat de(height, width, CV_8UC1);	//배열로 저장된 이미지의 픽셀 값들을 다시 Mat으로 변경
    for (int i = 0; i < size; i++) 
    {
        
        de.at<uchar>((i/width), (i%width)) = static_cast<uchar>(image_d[i]);
        
    }

//15. cca

    cv::Mat image_cc = cv::imread("images.png", cv::IMREAD_GRAYSCALE);
    int width_cca = image_cc.cols;
    int height_cca = image_cc.rows;
    int size_cca = width_cca * height_cca;

    int* image_cca = new int[size_cca];
    for (int i = 0; i < size_cca; i++)
    {
        image_cca[i] = image_cc.at<uchar>(i / width_cca, i % width_cca);
    }
    Thresh(image_cca,width_cca,  height_cca, 127);
    

    std::vector<std::vector<Point>> components;         //사이즈를 지금 정할 수 없어서 배열이 아닌 vector로 선언
    std::vector<Stat> stats;

    connectedComponentsWithStats(image_cca, width_cca, height_cca, components, stats);    //라벨 마다 속한 좌표들 components 벡터에 저장
                                                                                          //각 라벨의 사각형 값들 구조체 배열 stats에 저장
    int* image_ccaa = new int[size_cca];
    for (int i = 0; i < size_cca; i++)
    {
        image_ccaa[i] = image_cc.at<uchar>(i / width_cca, i % width_cca);
    }
    Thresh(image_ccaa,width_cca,  height_cca, 127);
    

    for (int i = 0; i < components.size(); i++)
    {
        rectangle(image_ccaa, stats[i].x -6, (stats[i].y -6), (stats[i].width+12), stats[i].height+12, width_cca);
    }

    for (int i = 0; i < components.size(); ++i)
    {
        std::cout << "Component " << i + 1 << ": ";
        for (int j = 0; j < components[i].size(); ++j)
        {
            std::cout << "(" << components[i][j].x << ", " << components[i][j].y << ") ";
        }
        std::cout << std::endl;
    }

    std::cout << "number of labeling is : " << components.size() << std::endl;

    cv::Mat g(height_cca, width_cca, CV_8UC1);
    for (int i = 0; i < size_cca; i++)
    {
        g.at<uchar>((i / width_cca), (i % width_cca)) = static_cast<uchar>(image_ccaa[i]);
    }

    cv::imshow("labeling", g);
    cv::waitKey(0);
    cv::destroyAllWindows();

    // 메모리 해제
    delete[] image_cca;
    delete[] image_ccaa;
    


    
   
 


    cv::imshow("delidation", de);


    cv::Mat im = cv::imread("lena.jpeg",cv::IMREAD_GRAYSCALE);
    cv::Mat img_canny_opencv(height, width, CV_8UC1);
    cv::Canny(im,img_canny_opencv,120,90);



    
    cv::imshow("canny_opencv", img_canny_opencv);

    cv::imshow("cal_canny",img_cal_canny);






    cv::Mat sobel;
    cv::Mat sobelX;
    cv::Mat sobelY;
    cv::Sobel(image_g, sobelX, CV_8U, 1, 0);
    cv::Sobel(image_g, sobelY, CV_8U, 0, 1);
    sobel = abs(sobelX) + abs(sobelY);


    cv::imshow("sobel", img_cal_sobel);
    cv::imshow("sobel_opencv", sobel);


	cv::Mat img_hsv_check;


	cv::cvtColor(image_c, img_hsv_check, cv::COLOR_BGR2HSV);


	
    
    cv::imshow("cal_blurring_aver",img_cal_blur_aver);
    cv::imshow("hsv", img_cal_hsv);
    cv::imshow("hsv_opencv",img_hsv_check);
	
	cv::waitKey(0);
	cv::destroyAllWindows();


    
    delete[] img1;

	
    return 0;
}



void reset(int*& image, int*& imgage_original, int& width, int& height)
{
    int size = width * height;
    
    delete[] image;

    image = new int[size];              
    

    for (int i = 0; i < size; i++) 
    {                                 
        
        image[i] = imgage_original[i];          //다른 함수로 인해 변경된 이미지를 원래의 이미지로 돌리기 위함
        
    }

}


void gogray(int*& color, int*& gray, int& width, int& height)
{
    int size = width*height;
    int sizeC = width * height *3;              //3차원 사이즈
    
    for (int i = 0; i < size; i++) 
    {                                 
        
            gray[i] = 0.114*color[i] + 0.587*color[i +size] + 0.299*color[i+ size+size];        //컬러 이미지를 grayscale로 변경하는 공식

    }


}

void rotateImage_90L(int*& image, int& width, int& height) 
{
    int newSize = height * width;
    int* rotatedImage_L = new int[newSize];

    for (int i = 0; i < newSize; i++) 
    {
        int h = static_cast<int>(i / width);
        int w = i % width ;
        int a = width -w -1;          //회전한 후의 행, 회전 후 열은 h, 회전 후 넓이는 width가 아닌 height

        rotatedImage_L[(a*height + h)] = image[i];          //픽셀들의 배열을 바꿔서 이미지를 왼쪽으로 회전
    }

    delete[] image;

    image = rotatedImage_L;

    // 너비와 높이 업데이트
    int temp = width;               //회전후 바탕과 이미지를 맞추기 위함
    width = height;
    height = temp;
}


void rotateImage_90R(int*& image, int& width, int& height) 
{
    int newSize = height * width;
    int* rotatedImage_R = new int[newSize];

    for (int i = 0; i < newSize; i++) 
    {
        int h = i / width;
        int w = i % width ;

        rotatedImage_R[(w * height) + (height - h)] = image[i];     //픽셀들의 배열을 바꿔서 이미지를 왼쪽으로 회전
    }                                                               //이미지의 한 픽셀 점을 기준으로 90도 왼쪽 회전한 위치의 좌표에 값을 지정

    delete[] image;

    image = rotatedImage_R;

    // 너비와 높이 업데이트
    int temp = width;
    width = height;
    height = temp;
}


void Thresh(int* image, int& width, int& height, int threshold) 
{     
    int size = width * height;
    for (int i = 0; i < size; i++) 
    {
        
            if (image[i] < threshold) 
            {                 //특정 값 이하는 검정색, 이상은 흰색으로 이진화 과정
                image[i] = 0;   
            } else 
            {
                image[i] = 255; 
            }
        
    }
}



void Thres_IVT(int* image, int& width, int& height, int threshold) 
{ 
        int size = width * height;
    //이진화 반전
    for (int i = 0; i < size; i++) 
    {
        
            if (image[i] < threshold) 
            {
                image[i] = 255;                         //특정 값 이상은 검정색, 이하는 흰색으로 이진화 과정
            } else 
            {
                image[i] = 0; 
            }
        
    }
}


void bgr_hsv(int*& img_color,double*& image_hsv , int width, int height)
{
    double max, min;

    int size = width * height;
    int sizeC = width * height * 3;
 

    
    for(int i=0; i < size; i++)
    {
        
            double b = img_color[i] / 255.0;
            double g = img_color[i+size] / 255.0;                   //행렬을 1차원으로 선언했기 때문에 두번 째 차원 배열은 +size, 세번 째 차원 배열은 +2*size
            double r = img_color[i + size + size] / 255.0;          //hsv

            double h;
            double s;
            double v;

            //bgr 색공간을 hsv 색공간으로 바꾸는 공식
            max = MAX(r,g);
            max = MAX(max,b);

            min = MIN(r,g);
            min = MIN(min,b);

            v = max;             
            double delta = max - min;                    
           
            if(delta ==0)
            {
                h =0;
                s =0;
            }
            else
            {
                s = delta / max;

                double r_d = (((max - r)/6) + (max/2)) / max;
                double g_d = (((max - g)/6) + (max/2)) / max;
                double b_d = (((max - b)/6) + (max/2)) / max;

                if(r ==max)
                {
                    h = b_d - g_d;
                }
                else if(g ==max)
                {
                    h = (1/3)+r_d - b_d;// 이미지 블러링 수행
                }
                else if(b==max)
                {
                    h = (2/3) + g_d - r_d;
                }
                h *= 60.0;

                if(h <0)
                {
                    h +=360;
                }
                if(h>1)
                {
                    h -= 360;
                }
                
            }

            

            
            

            image_hsv[i] = h*255.0;
            image_hsv[i+size] = s*255.0;            // !!중요 h,s,v 계산한 값은 0에서 1사이의 범위를 가짐
            image_hsv[i+size+size] = v*255.0;       // 그래서 다시 Mat출력을 위해서 255를 곱해서 반환
                                                     
            


  
            
        
    }
    }


void blur_aver(double* image, int width, int height, int kernelSize) 
{
    
    int size = width* height;
    double* blurredImage = new double[size];
    

    int kernelRadius = kernelSize / 2;  //소수점 버리고 정수로 저장
    

    
    for (int i = 0; i < height; ++i) 
    {
        for (int j = 0; j < width; ++j) 
        {
            double sum = 0.0;
            int count = 0;

            
            for (int k = -kernelRadius; k <= kernelRadius; ++k)         //j,i점에서 3*3 커널 연산
            {
                for (int l = -kernelRadius; l <= kernelRadius; ++l) 
                {
                    int x = j + l;
                    int y = i + k;

                        if(x >= 0 && x < width && y >= 0 && y < height) //j,i 가 0,0이거나 마지막 라인일때에 범위 벗어나는거 방지
                        {                                               //k,j for문안에 있으면 필요없으나 밖에서 blurredimage 계산해야함
                            sum += image[y*width + x];                  //한 픽셀 기준 주변의 모든 값들의 평균값을 기준 픽셀에 저장
                            ++count;
                        }
                }
                
            }

            blurredImage[i * width +j] = (sum / static_cast<double>(count));
        }
    }
    

    // 원본 이미지에 블러링 결과 복사
    for (int i = 0; i < size; ++i) 
    {
        
        image[i] = blurredImage[i];
        
    }

    // 임시 이미지 배열 메모리 해제
    
    delete[] blurredImage;
}


void blur_gausian(double* image, int width, int height) 
{
    // 블러링을 위한 임시 이미지 배열
    int size = width* height;
    double* out = new double[size];

    double mask[5][5] = { {1 / 256.,1 / 64.,6/256.,1 / 64.,1 / 256.},                   //정규분포 형태의 mask 합이 1
                          {1 / 64.,1/16.,24/256.0,1/16.,1 / 64.},
                          {6/256.,24/256.,36/256.,24/256.,6/256.},
                          {1 / 64.,1/16.,24/256.,1/16.,1 / 64.},
                          {1 / 256.,1 / 64.,6/256.,1 / 64.,1 / 256.} };
    

    

    // 이미지 블러링 수행
   for (int i = 0; i < height; i++) 
   {
        for (int j = 0; j < width; j++) 
        {
            double sum = 0.0;

            for (int m = -2; m < 3; m++) 
            {
                for (int n = -2; n < 3; n++) 
                {
                    sum += image[(i + m ) * width + (j + n )] * mask[m+2][n+2];
                }
            }

            out[i * width + j] = sum;
        }
    }
    

    // 원본 이미지에 블러링 결과 복사
    for (int i = 0; i < size; ++i) 
    {
        
        image[i] = out[i];
        
    }

    // 임시 이미지 배열 메모리 해제
    
    delete[] out;
}


void sobel(double* image, int& width, int& height) 
{

    int size = width * height;
    double* outH = new double[size];
    double* outV = new double[size];

    double maskV[3][3]              //수직 에지 검출을 위한 마스크
        = { {1,2,1} ,
            {0,0,0} ,
            {-1,-2,-1} };
    double maskH[3][3]              //수평 에지 검출을 위한 마스크
        = { {-1,0,1} ,
           {-2,0,2} ,
           {-1,0,1} };

    double S = 0.0; 
    for (int i = 0; i < height; i++) 
    {                                  
        for (int k = 0; k < width; k++) 
        {
            S = 0.0; 
            for (int m = 0; m < 3; m++) 
            {
                for (int n = 0; n < 3; n++) 
                {
                    S += image[(i + m)*width +k +n]* maskH[m][n];   
                }
            }
            outH[i*width +k] = S;
        }
    }
    S = 0.0; 
    for (int i = 0; i < height; i++) 
    {                                  
        for (int k = 0; k < width; k++) 
        {
            S = 0.0; 
            for (int m = 0; m < 3; m++) 
            {
                for (int n = 0; n < 3; n++) 
                {
                    S += image[(i + m)*width + k+n] * maskV[m][n];
                }
            }
            outV[i*width +k] = S;
        }
    }

    for (int i = 0; i < height; i++) 
    {
        for (int k = 0; k < width; k++) 
        {
            double v1 = outH[i*width +k];
            double v2 = outV[i*width +k];
            double v = v1 + v2;
            if (v > 255.0) v = 255.0;
            if (v < 0.0) v = 0.0;
            image[i*width +k] = (unsigned char)v;
        }
    }





    
}

void sharpenImage(double* image,int width,int height)
{
    int size = width*height;
    double sharpened[size];
    int kernel[9] = {-1, -1, -1,
                    -1,  9, -1,
                    -1, -1, -1};

    // 이미지 샤프닝 적용
    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            int sum = 0;
            for (int ky = 0; ky < 3; ky++) 
            {
                for (int kx = 0; kx < 3; kx++) 
                {
                    sum += image[(y + ky) * width + (x + kx)] * kernel[ky * 3 + kx];
                }
            }
            sharpened[y * width + x] = MIN(MAX(sum, 0), 255);
        }
    }
    for(int j=0;j<size;j++)
    {
        image[j]= sharpened[j];
    }

    
    
}



void canny(double* image_canny, int& width, int& height)
{
    

    
    int size = height*width;

    

    double  gradient[size];
    int direction[size];
    
    
    computeGradient(image_canny, gradient, direction, width, height);               //각 점에서의 기울기의 정도와 각도 구함

    double suppressed[size];
    nonMaximumSuppression(gradient, direction, suppressed, width, height);          //에지로 판단된 것들 중에서 필요없는 픽셀 삭제

    
    
    
    double edges[size];
    doubleThresholding(suppressed, edges, width, height, 90, 130);                  //임계값 두개를 이용한 이진화, 중간 값은 보류

    
    double outputImage[size];
    edgeTracking(edges, outputImage, width, height);                                //보류한 값들 근처에 에지 있으면 에지로 판단

    
    for(int i=0; i<size;i++)
    {
        image_canny[i] = outputImage[i];
    }




}


void computeGradient(double* inputImage, double* gradient, int* direction, int width, int height)
{
    int sobelX[9] = {-1, 0, 1,              //기울기 값을 구하기 위한 sobel
                    -2, 0, 2,               
                    -1, 0, 1};
    int sobelY[9] = {-1, -2, -1,
                    0, 0, 0,
                    1, 2, 1};

    int kernelSize = 3;
    

    int size = width * height;

    for (int i = 0; i < size; ++i) 
    {
        int x = i % width;
        int y = i / width;

        double gradientX = 0.0;
        double gradientY = 0.0;

        for (int k = 0; k < kernelSize; k++) 
        {
            for (int l = 0; l < kernelSize; l++) 
            {
                             
                    
                gradientX += inputImage[(y + k)*width +x+l] * sobelX[k * kernelSize + l];       //x,y 기울기 구해서 크기 구함
                gradientY += inputImage[(y + k)*width +x+l] * sobelY[k * kernelSize + l];
                
            }
        }

        gradient[i] = sqrt(gradientX * gradientX + gradientY * gradientY);  //기울기
        direction[i] = static_cast<int>(atan2(gradientY, gradientX) * 180.0 / M_PI);    //각도
    }
}

void nonMaximumSuppression(double* gradient, int* direction, double* suppressed, int width, int height) 
{
    int size = width * height;

    for (int i = 0; i < size; i++) 
    {
        suppressed[i] = gradient[i];                            //기울기로 이루어진 배열에서 에지가 아닌 것들 삭제
    }

    for (int y = 0; y < height; y++) 
    {
        for (int x = 0; x < width; x++) 
        {
            int index = (y * width + x);
            double angle = direction[index] % 180;
            if (angle < 0)
            {
                angle += 180;
            }

            int neighbor1, neighbor2;
            int weight1, weight2;
                                                                //각도를 그룹화 후 각자에서 비교, 
            
            if ((angle >= 0.0 && angle < 22.5) || (angle >= 157.5 && angle <= 180.0))
            {
                neighbor1 = index - width;

                neighbor2 = index + width;
                weight1 = abs(gradient[neighbor1] - gradient[index]);   //이웃1에 대한 가중치
                weight2 = abs(gradient[neighbor2] - gradient[index]);   //이웃2에 대한 가중치
            } 
            else if (angle >= 22.5 && angle < 67.5) 
            {
                neighbor1 = index - width + 1;
                neighbor2 = index + width - 1;
                weight1 = abs(gradient[neighbor1] - gradient[index]);
                weight2 = abs(gradient[neighbor2] - gradient[index]);
            } 
            else if (angle >= 67.5 && angle < 112.5) 
            {
                neighbor1 = index - 1;
                neighbor2 = index + 1;
                weight1 = abs(gradient[neighbor1] - gradient[index]);
                weight2 = abs(gradient[neighbor2] - gradient[index]);
            } 
            else if(angle >=112.5 && angle <157.5) 
            {
                neighbor1 = index - width - 1;
                neighbor2 = index + width + 1;
                weight1 = abs(gradient[neighbor1] - gradient[index]);
                weight2 = abs(gradient[neighbor2] - gradient[index]);
            }

            
            if (gradient[index] <= weight1 || gradient[index] <= weight2)     //가중치보다 크면 에지로 인정되지만 둘중 하나라도 아닌 경우에는 에지 탈락
            {                                                                 //에지 강도가 주변(가중치) 보다 낮으면 에지 아님
                suppressed[index] = 0;
            }
        }
    }
}

void doubleThresholding(double* suppressed, double* edges, int width, int height, int lowThreshold, int highThreshold) 
{
    int size = width * height;                  //임계값 두개, 이상은 에지 확정 이하는 에지 아님, 사이는 128로 보류

    for (int i = 0; i < size; ++i) 
    {
        if (suppressed[i] >= highThreshold) 
        {
            edges[i] = 255;
        } else if (suppressed[i] >= lowThreshold && suppressed[i] < highThreshold) 
        {
            edges[i] = 128;
        } else 
        {
            edges[i] = 0;
        }
    }
}

void edgeTracking(double* edges, double* output, int width, int height) 
{
    int size = width * height;                  //128로 보류한 것들중에서 한 픽셀 중심 기준으로 근처에 에지가 있으면 에지로 인정

    for (int i = 0; i < size; ++i) 
    {
        output[i] = edges[i];

        if (edges[i] == 128) 
        {
            
            if (i - width - 1 >= 0 && edges[i - width - 1] == 255) 
            {
                output[i] = 255;
            } else if (i - width >= 0 && edges[i - width] == 255) 
            {
                output[i] = 255;
            } else if (i - width + 1 >= 0 && edges[i - width + 1] == 255) 
            {
                output[i] = 255;
            } else if (i - 1 >= 0 && edges[i - 1] == 255) 
            {
                output[i] = 255;
            } else if (i + 1 < size && edges[i + 1] == 255) 
            {
                output[i] = 255;
            } else if (i + width - 1 < size && edges[i + width - 1] == 255) 
            {
                output[i] = 255;
            } else if (i + width < size && edges[i + width] == 255) 
            {
                output[i] = 255;
            } else if (i + width + 1 < size && edges[i + width + 1] == 255) 
            {
                output[i] = 255;
            } else 
            {
                output[i] = 0;
            }
        }
    }
}


void erosion(int* image, int width, int height) 
{
    int kernel[3][3] = {{1, 1, 1},
                        {1, 1, 1},
                        {1, 1, 1}};

    int erodedImage[width * height];

    
    for (int i = 1; i < height-1; i++) 
    {
        for (int j = 1; j < width-1; j++) 
        {
            bool background = true;

            
            for (int m = -1; m < 2; m++)                     //image의 j,i에서 커널 3*3만큼의 픽셀들을 다 조사해서 하나라도 0이라면 false
            {                                               //가장자리는 끝부분이 배경이므로 가장자리 픽셀값을 0으로 만들수 있음
                for (int n = -1; n < 2; n++) 
                {
                    if (image[(i-1 + m) * width + (j-1 + n)] == 0) 
                    {
                        background = false;
                        
                    }
                
                }

            
                if (!background)                            //해당 j,i에서 0있어서 false 나오면 출력 이미지의 해당 j,i는 0 아니면 255
                {
                    erodedImage[(i-1) * width + j-1] = 0;   //근처에 까만 픽셀이 하나라도 있으면 해당 픽셀 까만색으로 변경
                } else 
                {
                    erodedImage[(i-1) * width + j-1] = 255;
                }
            }
        }
    }

    
    for (int i = 0; i < height; ++i) 
    {
        for (int j = 0; j < width; ++j) 
        {
            image[i * width + j] = erodedImage[i * width + j];
        }
    }

    
}

void delitation(int* im, int width, int height) 
{
    int kernel[3][3] = {{1, 1, 1},
                        {1, 1, 1},
                        {1, 1, 1}};

    

    
    for (int i = 1; i < height-1; i++)                  
    {
        for (int j = 1; j < width-1; j++)                       //커널 범위가 -1부터 1까지이므로 0번째를 보기위해서 1부터 시작
        {
            
            bool white = false;
            
            for (int m = -1; m < 2; m++)                       //픽셀 기준으로 왼쪽, 오른쪽 다 확인해야하므로 -1부터 1까지의 커널 범위
            {                                               
                for (int n = -1; n < 2; n++) 
                {
                    if(im[(i + m) * width + (j + n)] == 255)    //근처에 흰색이 하나라도 있으면 해당 픽셀 흰색으로
                    {
                        white = true;
                    }   
                }
            }
            if(white){
                im[width*(i-1)+j-1] = 255;
            }
        }
    }      
}




void erosion_double(double* image, int width, int height) 
{
    int kernel[3][3] = {{1, 1, 1},
                        {1, 1, 1},
                        {1, 1, 1}};

    double erodedImage[width * height];

    
    for (int i = 1; i < height-1; i++) 
    {
        for (int j = 1; j < width-1; j++) 
        {
            bool background = true;

            
            for (int m = -1; m < 2; m++)                     //image의 j,i에서 커널 3*3만큼의 픽셀들을 다 조사해서 하나라도 0이라면 false
            {                                               //가장자리는 끝부분이 배경이므로 가장자리 픽셀값을 0으로 만들수 있음
                for (int n = -1; n < 2; n++) 
                {
                    if (image[(i-1 + m) * width + (j-1 + n)] == 0) 
                    {
                        background = false;
                        
                    }
                
                }

            
                if (!background)                            //해당 j,i에서 0있어서 false 나오면 출력 이미지의 해당 j,i는 0 아니면 255
                {
                    erodedImage[(i-1) * width + j-1] = 0;   //근처에 까만 픽셀이 하나라도 있으면 해당 픽셀 까만색으로 변경
                } else 
                {
                    erodedImage[(i-1) * width + j-1] = 255;
                }
            }
        }
    }

    
    for (int i = 0; i < height; ++i) 
    {
        for (int j = 0; j < width; ++j) 
        {
            image[i * width + j] = erodedImage[i * width + j];
        }
    }

    
}



void rectangle(int* image, int x, int y, int width, int height, int imageWidth)
{
    for(int i = 0; i < width; i++)
    {
        image[imageWidth*y+(i+x)] = 255;
        image[imageWidth*(y+height-1) + i+x] = 255;
    }
    for(int j=0; j<height;j++)
    {
        image[imageWidth*(j+y)+x] = 255;
        image[imageWidth*(j+y) + x+ width-1] = 255;
    }
}



void dfs(int x, int y, int label, int* image, int width, int height, Point* points, int& pointIndex)  //재귀함수, 한점과 근처에 연결된 점 중에서 255가 있으면 없을때 까지 계속 진행 (4방향)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
        return;                                             //반복 중에 범위 벗어나면 중지

    int index = y * width + x;
    if (image[index] != 255)
        return;                                             //근처에 255픽셀이 없으면 중지

    image[index] = label;                                   

    Point p;
    p.x = x;
    p.y = y;
    points[pointIndex] = p;                                 //원하는 픽셀의 정보를 구조체 배열인 Points에 저장
    pointIndex++;

    dfs(x - 1, y, label, image, width, height, points, pointIndex); 
    dfs(x + 1, y, label, image, width, height, points, pointIndex); //4방향으로 반복해서 쭉 뻗어나가다가 위 두조건에 걸리면 정지
    dfs(x, y - 1, label, image, width, height, points, pointIndex); 
    dfs(x, y + 1, label, image, width, height, points, pointIndex); 
}

void connectedComponentsWithStats(int* image, int width, int height, std::vector<std::vector<Point>>& components, std::vector<Stat>& stats)
{
    int label = 1; // 레이블 초기값

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int index = y * width + x;

            int x_max = 0;
            int x_min = width;          //이 값보다 작으면 그 값이 최소이므로 초기 값을 최대인 넓이로 설정
            int y_max = 0;
            int y_min = height;

            if (image[index] == 255)
            {
                Point* points = new Point[width * height];
                int pointIndex = 0;

                dfs(x, y, label, image, width, height, points, pointIndex); // 해당 점과 연결된 픽셀의 좌표의 배열과 해당 번호의 물체의 픽셀수 반환

                for (int i = 0; i < pointIndex; i++)
                {
                    int px = points[i].x;           //사각형을 그리기 위해 x,y의 최대,최소 구함
                    int py = points[i].y;

                    if (px > x_max)
                        x_max = px;
                    if (px < x_min)
                        x_min = px;
                    if (py > y_max)
                        y_max = py;
                    if (py < y_min)
                        y_min = py;
                }

                Stat s;                             //구조체로 첫번 째 물체의 정보를 s 첫번째 원소에 저장...
                s.x = x_min;
                s.y = y_min;
                s.width = x_max - x_min + 1;
                s.height = y_max - y_min + 1;
                s.area = pointIndex;

                stats.push_back(s);

                std::vector<Point> component(points, points + pointIndex);
                components.push_back(component);

                delete[] points;

                label++;
            }
        }
    }
}