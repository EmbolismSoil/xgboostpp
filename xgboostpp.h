#ifndef __XGBOOSTPP_H__
#define __XGBOOSTPP_H__

#include <string>
#include <xgboost/c_api.h>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>

template<class T>
class Features
{
public:
    Features(uint64_t const nrow, uint64_t ncol):
        _ncol(ncol),
        _nrow(nrow)
    {
        _data = static_cast<T*>(std::malloc(sizeof(T)*_ncol*_nrow));
        if (!_data){
            //LOG HERE
            _data = nullptr;
        }     

        T initvalue(0);
        for (auto i = 0; i < _ncol*_nrow; ++i){
            _data[i] = initvalue;
        }
    }

    T* row(uint32_t n)
    {
        if (_nrow <= n){
            //LOG HERE
            return nullptr;
        }

        if (!_data){
            //LOG HERE
            return nullptr;
        }

        return _data + n*_ncol;
    }

    uint64_t nrow() const
    {
        return _nrow;
    }

    uint64_t ncols() const
    {
        return _ncol;
    }

    const T* data(void) const
    {
        return _data;
    }

    virtual ~Features()
    {
        if (_data){
            std::free(static_cast<void*>(_data));
        }
    }

private:
    uint64_t const _ncol;
    uint64_t const _nrow;
    T* _data;
};


class XGBoostPP
{
public:
    XGBoostPP(std::string const& path, uint64_t const ncol, uint64_t nlabels):
        _modelPath(path),
        _ncol(ncol),
        _nlabels(nlabels)
    {
       
        if (XGBoosterCreate(NULL, 0, &_booster) == 0 &&  XGBoosterLoadModel(_booster, _modelPath.c_str()) == 0){
            //LOG HERE
        }else{
            //LOG HERE
            _booster = NULL;
        }        
    }

    int predict(std::shared_ptr<Features<float>> const features, std::vector<std::vector<float>>& vec)
    {
        DMatrixHandle X;
        const float* data = features->data();
        auto nrow = features->nrow();

        XGDMatrixCreateFromMat(data, nrow, _ncol, NAN, &X);
        
        const float* out;
        uint64_t l;
        auto ret = XGBoosterPredict(_booster, X, 0, 0,  &l, &out);
        if (ret < 0){
            // LOG HERE
            return -1;
        }

        std::vector<float> tmp;
        std::copy(out, out + nrow*_nlabels, std::back_inserter(tmp));

        XGDMatrixFree(X);
        
        if (l != nrow*_nlabels){
            //LOG HERE
            return -1;
        }
        
        for (auto i = 0; i < nrow*_nlabels; i += _nlabels){
            std::vector<float> r;
            std::copy(out+i, out+i+_nlabels, std::back_inserter(r));
            vec.emplace_back(std::move(r));
        }

        return 0;
    }

    std::shared_ptr<Features<float>> allocFeatures(uint64_t nrow)
    {
        auto features = std::make_shared<Features<float>>(nrow, _ncol);
        if (!features->data()){
            return nullptr;
        }

        return features;
    }

    virtual ~XGBoostPP(){
        XGBoosterFree(_booster);
    }
    
private:
    std::string const _modelPath;
    BoosterHandle _booster;
    uint64_t const _ncol;
    uint64_t const _nlabels;
};

#endif