#include "neural_net.hpp"
//------------------------
// Constructor
//------------------------

NeuralNet::NeuralNet(float const& mRate, float const& cRate, std::vector<float> const& dims) {
    this->mRate = mRate;
    this->cRate = cRate;
    this->numCrossovers = 0;
    this->numMutations = 0;
    this->inputSize = int(dims[0]);
    this->outputSize = int(dims[dims.size()-1]);
    for (int i = 0; i < dims.size()-1; i++) {
        // he initialization
        const float HE = (1.0f / sqrtf(dims[i+1]/2.0f));

        W.push_back(MatrixXf::Random(int(dims[i+1]), int(dims[i]+0.1)) * HE);
        b.push_back(VectorXf::Zero(int(dims[i+1]+0.1)));
    }
}

//------------------------
// Activation Functions
//------------------------

float NeuralNet::relu(float const& F) {
    return (F > 0) ? F : 0;
}

VectorXf NeuralNet::relu(VectorXf const& V) {
    VectorXf A(V.rows());
    
    for (int r = 0; r < V.rows(); r++) {
        A[r] = relu(V(r));
    }
    
    return A;
}

float NeuralNet::sigmoid(float const& F) {
    return 1.0f / (1.0f + expf(-F));
}

VectorXf NeuralNet::sigmoid(VectorXf const& V){
    VectorXf A(V.rows());
    
    for (int r = 0; r < V.rows(); r++) {
        A[r] = sigmoid((float)V(r));
    }
    
    return A;
}

VectorXf NeuralNet::softmax(VectorXf const& V) {
    VectorXf A(V.rows());
    float sum = 0;
    
    for (int r = 0; r < V.rows(); r++) {
        A(r) = expf(V(r));
        sum += A(r);
    }
    
    for (int r = 0; r < V.rows(); r++) {
        A(r) = A(r) / sum;
    }
    
    return A;
}

//----------------------------
// Genetic Algorithm Methods
//----------------------------

std::unique_ptr<Mutable> NeuralNet::clone() const {
    return std::unique_ptr<Mutable>(new NeuralNet(*this));
}

std::vector<float> NeuralNet::predict(std::vector<float> const& X) const {
    const int L = W.size(); // num layers
    
    VectorXf A(X.size());
    for (int i = 0; i < X.size(); i++) {
        A(i) = X[i];
    }
    
    for (int l = 0; l < L-1; l++) {
        VectorXf Z = W[l] * A;
        Z += b[l];
        A = relu(Z);
    }
    
    VectorXf Z = W[L-1] * A;
    Z.colwise() += b[L-1];
    A = softmax(Z);
    
    return std::vector<float>(A.begin(), A.end());
}

void NeuralNet::mutate() {
    for (int i = 0; i < W.size(); i++) {
        // find the mutation weights
        if (randFloat() < mRate) {
            const float HE =  (float(NEURAL_NET_MUTATION_MAG) / sqrtf(W[i].rows()/2.0f));
            W[i] += MatrixXf::Random(W[i].rows(), W[i].cols()) * HE;
            b[i] += VectorXf::Random(b[i].rows()) * HE;
        }
    }
    
    numMutations++;
}
    
void NeuralNet::crossover(Mutable const& otherN) {
    NeuralNet& other = (NeuralNet&) otherN;
    const float avgCRate = (cRate + other.cRate)/2;
    const float avgMRate = (mRate + other.mRate)/2;
    const float chanceOther = (1.0f-avgCRate) / 2.0f;
    
    for (int i = 0; i < W.size(); i++) {
        
        for (int r = 0; r < W[i].rows(); r++) {
            float crossoverChance = randFloat();
            if (crossoverChance <= avgCRate) {
                int randVal = rand();
                int crossInd = randVal % W[i].cols();
                b[i](r) = (randVal%2) ? b[i](r) : other.b[i](r);
            
                for (int c = 0; c < W[i].cols(); c++) {
                    W[i](r, c) = (c < crossInd) ? W[i](r, c) : other.W[i](r, c);
                }
            } else if (crossoverChance <= chanceOther + avgCRate) {
                // get the other's values instead
                W[i] = other.W[i];
                b[i] = other.b[i];
            }
        }
    }
    
    numCrossovers++;
}