// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/torch.h>
#include <vector>
#include <algorithm>

namespace scheduler {
template<typename TOptimizer>
struct OptimizerOptionsMap {
};

template<>
struct OptimizerOptionsMap<torch::optim::Adam> {
    using type = torch::optim::AdamOptions;
};

template<>
struct OptimizerOptionsMap<torch::optim::Adagrad> {
    using type = torch::optim::AdagradOptions;
};

template<>
struct OptimizerOptionsMap<torch::optim::LBFGS> {
    using type = torch::optim::LBFGSOptions;
};

template<>
struct OptimizerOptionsMap<torch::optim::RMSprop> {
    using type = torch::optim::RMSpropOptions;
};

template<>
struct OptimizerOptionsMap<torch::optim::SGD> {
    using type = torch::optim::SGDOptions;
};

/**
 * Learning rate scheduler base.
 *
 * Based on the Python implementation at
 * https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.
 * @tparam TOptimizer Optimizer type
 */
template<typename TOptimizer>
class LRScheduler {
 public:
    explicit LRScheduler(TOptimizer& optimizer, int64_t last_epoch = -1)
            : optimizer_(optimizer), last_epoch_(last_epoch), base_lrs(get_current_lr()) {}

    virtual std::vector<double> get_lr() = 0;

    void step() {
        ++last_epoch_;

        const auto values = get_lr();
        auto &param_groups = optimizer_.param_groups();

        for (decltype(param_groups.size()) i = 0; i != param_groups.size(); ++i) {
            dynamic_cast<typename OptimizerOptionsMap<TOptimizer>::type &>(param_groups[i].options()).lr(values[i]);
        }
    }

    virtual ~LRScheduler() = default;

 protected:
    TOptimizer& optimizer_;
    int64_t last_epoch_;
    std::vector<double> base_lrs;

    std::vector<double> get_current_lr() {
        std::vector<double> lrs;
        lrs.reserve(optimizer_.param_groups().size());

        for (auto &param_group : optimizer_.param_groups()) {
            lrs.push_back(dynamic_cast<typename
            OptimizerOptionsMap<TOptimizer>::type &>(param_group.options()).lr());
        }

        return lrs;
    }
};

/**
 * Step learning rate scheduler.
 *
 * Based on the python implementation at
 * https://github.com/pytorch/pytorch/blob/master/torch/optim/lr_scheduler.py.
 * @tparam TOptimizer Optimizer type
 */
template<typename TOptimizer>
class StepLR : public LRScheduler<TOptimizer> {
 public:
    StepLR(TOptimizer& optimizer, int64_t step_size, double gamma = 0.1, int64_t last_epoch = -1)
            : LRScheduler<TOptimizer>(optimizer, last_epoch), step_size_(step_size), gamma_(gamma) {}

    std::vector<double> get_lr() override {
        auto new_lr = this->get_current_lr();

        if (this->last_epoch_ != 0 && (this->last_epoch_ % step_size_ == 0)) {
            std::transform(new_lr.cbegin(), new_lr.cend(), new_lr.begin(),
                           [gamma_ = gamma_](auto value) { return value * gamma_; });
        }

        return new_lr;
    }

 private:
    int64_t step_size_;
    double gamma_;
};
}  // namespace scheduler