#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>

// TODO: Template this
//template <size_t T>
Eigen::TensorFixedSize<double, Eigen::Sizes<2, 10, 3, 2, 9>> helicityWeightsToTensor(ROOT::VecOps::RVec<double> weights, double ptV, double charge) {
    std::array<double, 11> ptVbins = {0.0, 2.9, 4.7, 6.7, 9.0, 11.8, 15.3, 20.1, 27.2, 40.2, 13000.0};
    size_t ptbin = std::distance(std::begin(ptVbins), std::lower_bound(std::begin(ptVbins), std::end(ptVbins), ptV)) - 1;
    size_t chargebin = charge < 0 ? 0 : 1;
    Eigen::TensorFixedSize<double, Eigen::Sizes<2, 10, 3, 2, 9>> out;
    out.setConstant(1.);

    for (size_t i = 0; i < weights.size(); i++) {
        size_t variation = 2; // muF
        const size_t nparams = 9;
        const size_t nvars = 3;

        if ((i >= 0 && i < nparams) || (i >= nparams*nvars && i < nparams*(nvars+1)))
            variation = 0;
        else if ((i >= nparams && i < 2*nparams) || (i >= nparams*(nvars+1) && i < nparams*(nvars+2)))
            variation = 1;
        // Down = 0, up = 1
        size_t shifttype = i < nparams*nvars ? 1 : 0;
        out(chargebin, ptbin, variation, shifttype, i % nparams) = weights[i];
        std::cout << "Filling ptbin " << ptbin << " with value " << weights[i] << std::endl;
    }
    return out;
}
