
#include "xgboostpp.h"
#include <algorithm>
#include <iostream>

int main(int argc, const char* argv[])
{
    auto nsamples = 2;
    auto xgb = XGBoostPP(argv[1], 4, 3); //特征列有4列, label有3个, iris例子中分别为三种类型的花，回归任何的话，这里nlabel=1即可

    float feats[] = 
    {
        4.6, 3.4, 1.4, 0.3,
        6.2, 2.9, 4.3, 1.3
    };
    
    Eigen::MatrixXf features = Eigen::Map<Eigen::MatrixXf>(static_cast<float*>(feats), 2, 4);
    Eigen::MatrixXf y;
    auto ret = xgb.predict(features, y);
    if (ret != 0){
        std::cout << "predict error" << std::endl;
    }
    
    std::cout << y << std::endl;
}
