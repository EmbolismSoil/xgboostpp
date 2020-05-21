
#include "xgboostpp.h"
#include <algorithm>
#include <iostream>

int main(int argc, const char* argv[])
{
    auto nsamples = 2;
    auto xgb = XGBoostPP(argv[1], 4, 3); //特征列有4列, label有3个, iris例子中分别为三种类型的花，回归任何的话，这里nlabel=1即可

    //result = array([[9.9658281e-01, 2.4966884e-03, 9.2058454e-04],
    //       [9.9608469e-01, 2.4954407e-03, 1.4198524e-03]], dtype=float32)
    //
    float feats[] = 
    {
        5.1, 3.5, 1.4, 0.2,
        4.9, 3.0, 1.4, 0.2
    };
    
    Eigen::MatrixXf features; 
    XGBoostPP::vector2Matrix(features, feats, 2, 4);
    Eigen::MatrixXf y;
    auto ret = xgb.predict(features.transpose(), y);
    if (ret != 0){
        std::cout << "predict error" << std::endl;
    }
    
    std::cout << "intput : \n" << features << std::endl << "output: \n" << y << std::endl;
}
