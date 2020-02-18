#include "xgboostpp.h"
#include <algorithm>
#include <iostream>

int main(int argc, const char* argv[])
{
    auto nsamples = 2;
    auto xgb = XGBoostPP(argv[1], 4, 3); //特征列有4列, label有3个
    auto features = xgb.allocFeatures(nsamples);//分配5行特征储存空间

    float feats[][4] = 
    {
        {4.6, 3.4, 1.4, 0.3},
        {6.2, 2.9, 4.3, 1.3}
    };
    
    for (auto i = 0; i < nsamples; ++i){
        auto f = features->row(i); //这里f1可以看成是一个长度为4的数组
        volatile float tmp[4];
        for (auto j = 0; j < 4; ++j){
            f[j] = feats[i][j];
        }
    }


    std::vector<std::vector<float>> prob;
    auto ret = xgb.predict(features, prob);
    if (ret != 0){
        std::cout << "predict error" << std::endl;
    }

    for (auto const& item: prob){
        for (auto const& n: item){
            std::cout << n << ", ";
        }
        std::cout << std::endl;
    }

}