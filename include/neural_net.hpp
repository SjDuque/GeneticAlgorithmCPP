#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include "mutable.hpp"
#include <Eigen/Dense>

#define NEURAL_NET_MUTATION_MAG 0.2

using namespace Eigen;

/**
 * @brief Fully-Connected Neural Network
 * 
 */
class NeuralNet : public Mutable {
    
    private:
    
    std::vector<MatrixXf> W; // weight matrices
    std::vector<VectorXf> b; // bias vectors
    
    public:
    
    /**
     * @brief Construct an empty new Network object
     * 
     */
    NeuralNet() {}
    virtual ~NeuralNet() {}
    NeuralNet(NeuralNet const& base) = default;
    NeuralNet(std::unique_ptr<Mutable> const& other) : NeuralNet(*(NeuralNet *)other.get()) { };
    
    
    /**
     * @brief Construct a new Network object
     * 
     * @param DIMS Specify the dimensions of each layer.
     * 
     * The length represents the number of layers, with
     * the first representing the input and the last representing
     * the output. 
     */
    NeuralNet(float const& mRate, float const& cRate, std::vector<float> const& params);
    
    NeuralNet& operator=(NeuralNet const& other) = default;
    
    NeuralNet& operator=(std::unique_ptr<Mutable> const& other) {
        return operator=(*(NeuralNet *)other.get());
    }
        
    // ACTIVATION FUNCTIONS

    static float relu(const float& F);
    static VectorXf relu(const VectorXf& V);
    static MatrixXf relu(const MatrixXf& M);
    
    static float sigmoid(const float& F);
    static VectorXf sigmoid(const VectorXf& V);
    static MatrixXf sigmoid(const MatrixXf& X);
    
    static VectorXf softmax(const VectorXf& Z);
    static MatrixXf softmax(const MatrixXf& M);
    
    // OPERATIONS
    
    virtual std::unique_ptr<Mutable> clone() const;
    virtual std::vector<float> predict(std::vector<float> const& X) const;
    virtual void mutate();
    virtual void crossover(Mutable const& other);
};

#endif