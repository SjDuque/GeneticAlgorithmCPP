#ifndef CHROMOSOME_HPP
#define CHROMOSOME_HPP

#include "mutable.hpp"

#define CHROMOSOME_MUTATION_MAG 0.2
#define MUTATION_INIT_ALPHA 0.1

template<typename T>
class Chromosome : public Mutable {
    static_assert(std::is_base_of<Mutable, T>::value, "Chromosome needs typename Mutable");
    
    private:
    std::vector<T> genes;
    std::vector<float> geneWeights;
    
    public:
    
    Chromosome() {}
    virtual ~Chromosome() {}
    
    Chromosome(Chromosome const& other) = default;
    Chromosome(std::unique_ptr<Mutable> const& other) : Chromosome(*(Chromosome *)other) { };
    
    /**
     * @brief Construct a new Chromosome object
     * 
     * @param mRate Mutation rate, range: [0.0, 1.0]
     * @param cRate Crossover rate, range: [0.0, 1.0]
     * @param params params[0] is the number of genes,
     *               remaining values are for the T constructor besides mRate and cRate
     * 
     */
    Chromosome(float const& mRate, float const& cRate, std::vector<float> const& params) {
        this->mRate = mRate;
        this->cRate = cRate;
        this->numCrossovers = 0;
        this->numMutations = 0;
        
        const int size = int(params[0]);
        genes.reserve(size);
        geneWeights.reserve(size);
        
        float upper = 1.0f;
        for (int i = 0; i < size; i++) {
            float lower = 1.0f - (1.0f / float(size))*i;
            float childMRate = randFloat() * (upper - lower) + lower;
            float childCRate = randFloat() * (upper - lower) + lower;
            
            genes.push_back(T(childMRate, childCRate, std::vector<float>(params.begin()+1, params.end())));
            geneWeights.push_back(randFloat());
            
            upper = lower;
        }
        
        this->inputSize = genes[0].getInputSize();
        this->outputSize = genes[0].getOutputSize();
        
        limit();
    }
    
    Chromosome& operator=(Chromosome const& other) = default;
    
    Chromosome& operator=(std::unique_ptr<Mutable> const& other) {
        return operator=(*(Chromosome *)other.get());
    }
    
    static void addVector(std::vector<float> & A, std::vector<float> const& B, float const& weight) {
        for (int i = 0; i < A.size(); i++) {
            A[i] += B[i] * weight;
        }
    }
    
    static float limit(float const& x) {
        return fmin(fmax(x, 0.0f), 1.0f);
    }
    
    void limit() {
        mRate = limit(mRate);
        cRate = limit(cRate);
        softmax(geneWeights);
    }
    
    virtual std::unique_ptr<Mutable> clone() const {
        return std::unique_ptr<Mutable>(new Chromosome(*this));
    }

    virtual std::vector<float> predict(std::vector<float> const& X) const {
        std::vector<float> sum(getOutputSize(), 0.0f);
        
        for (int i = 0; i < genes.size(); i++) {
            addVector(sum, genes[i].predict(X), geneWeights[i]);
        }
        
        softmax(sum);
        return sum;
    }
    
    virtual void mutate() {
        for (int i = 0; i < genes.size(); i++) {
            if (randFloat() < mRate) {
                genes[i].mutate();
                geneWeights[i] += randFloat() * float(CHROMOSOME_MUTATION_MAG);
            }
        }
        
        if (randFloat() < mRate) {
            mRate += randFloat() * mRate * float(CHROMOSOME_MUTATION_MAG);
        }
        
        if (randFloat() < mRate) {
            cRate += randFloat() * cRate * float(CHROMOSOME_MUTATION_MAG);
        }
        
        limit();
        numMutations++;
    }
    
    virtual void crossover(Mutable const& otherC) {
        Chromosome<T>& other = (Chromosome<T>&) otherC;
        const float avgCRate = (cRate + other.cRate)/2;
        const float avgMRate = (mRate + other.mRate)/2;
        const float chanceOther = (1.0f-avgCRate) / 2.0f;
        
        for (int i = 0; i < genes.size(); i++) {
            genes[i].crossover(other.genes[i]);
            
            float crossoverChance = randFloat();
            if (crossoverChance < avgCRate) {
                geneWeights[i] = (geneWeights[i] + other.geneWeights[i])/2;
            } else if (crossoverChance < chanceOther + avgCRate) {
                // get the other's values instead
                geneWeights[i] = other.geneWeights[i];
            }
        }
        
        float crossoverChance = randFloat();
        if (crossoverChance < avgCRate) {
            cRate = avgCRate;
        } else if (crossoverChance < chanceOther + avgCRate) {
            cRate = other.cRate;
        }
        
        crossoverChance = randFloat();
        if (crossoverChance < avgCRate) {
            mRate = avgMRate;
        } else if (crossoverChance < chanceOther + avgCRate) {
            mRate = other.mRate;
        }
        
        limit();
        numCrossovers++;
    }
};

#endif