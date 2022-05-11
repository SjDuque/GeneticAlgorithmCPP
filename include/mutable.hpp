#ifndef MUTABLE_HPP
#define MUTABLE_HPP

#include <memory>
#include <vector>
#include <cmath>
#include <random>

/**
 * @brief Abstract Mutable class
 * 
 */
class Mutable {    
    protected:
    float mRate; // mutation rate
    float cRate; // crossover rate
    int inputSize;
    int outputSize;
    int numMutations;
    int numCrossovers;
    
    public:
    
    virtual ~Mutable() {}
    
    static void softmax(std::vector<float> & A) {
        float weightSum = 0;
        for (int i = 0; i < A.size(); i++) {
            A[i] = expf(A[i]);
            weightSum += A[i];
        }
        
        // perform a softmax
        for (int i = 0; i < A.size(); i++) {
            A[i] /= weightSum;
        }
    }
    
    static float randFloat() {
        return float(rand()) / float(RAND_MAX);
    }
    
    virtual float getMRate() const { return mRate; }
    virtual float getCRate() const { return cRate; }
    
    virtual float getInputSize() const { return inputSize; }
    virtual float getOutputSize() const { return outputSize; }
    
    virtual int getNumMutations() const { return numMutations; }
    virtual int getNumCrossovers() const { return numCrossovers; }
    
    virtual std::unique_ptr<Mutable> clone() const=0;
    virtual std::vector<float> predict(std::vector<float> const& X) const=0;
    virtual void mutate()=0;
    virtual void crossover(Mutable const& other)=0;
};

#endif