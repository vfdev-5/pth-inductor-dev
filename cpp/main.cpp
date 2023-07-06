
#include <iostream>
#include <stdlib.h>
#include <iomanip>


int main() {

    std::cout << "---" << std::endl;

    long i1 = 31;
    // j = torch.arange(out_w, dtype=input.dtype, device=input.device)
    auto tmp1 = static_cast<long>(i1);
    auto tmp2 = static_cast<double>(tmp1);
    auto tmp3 = static_cast<double>(1.0);
    auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
    auto tmp5 = static_cast<double>(0.0);
    auto tmp6 = tmp4 + tmp5;

    // y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)
    auto tmp7 = static_cast<float>(tmp6);
    auto tmp8 = static_cast<float>(0.5);
    auto tmp9 = tmp7 + tmp8;
    auto tmp10 = static_cast<float>(2.0357142857142856);
    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
    auto tmp12 = tmp11 - tmp8;

    std::cout << "tmp12: " << std::fixed << std::setprecision(8)
              << tmp12 << std::endl;

    return 0;
}