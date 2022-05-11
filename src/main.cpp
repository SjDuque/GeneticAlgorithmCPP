#include <iostream>
#include <vector>
#include "neural_net.hpp"
#include "chromosome.hpp"

using namespace std;

int main(void) {
    srand(time(NULL));
    // NeuralNet parent1(0.25, 0.4, {5, 4, 10, 3});
    // NeuralNet parent2(0.3, 0.3, {5, 4, 10, 3});
    Chromosome<NeuralNet> parent1(0.25, 0.5, {100, 5, 10, 15, 4});
    Chromosome<NeuralNet> parent2(0.5, 0.25, {100, 5, 10, 15, 4});
    std::vector<float> pred;
    Chromosome<NeuralNet> temp;
    // base = *(Chromosome<NeuralNet> *)(base.clone().get());
    std::vector<float> X = {0.2, 0.5, 0.2, 0.3, 0.8};
    
    cout << "X input" << endl;
    copy(X.begin(), X.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    cout << "testing prediction" << endl;
    pred = parent1.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    pred = parent2.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    cout << "testing crossover: " << endl;
    temp = parent1.clone();
    temp.crossover(parent2);
    pred = temp.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    temp = parent2.clone();
    temp.crossover(parent1);
    pred = temp.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    cout << "testing mutate: " << endl;
    temp = parent1.clone();
    temp.mutate();
    pred = temp.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    temp = parent2.clone();
    temp.mutate();
    pred = temp.predict(X);
    copy(pred.begin(), pred.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
    
    // cout << copy(base.begin(), base.end(), ostream_iterator<float>(cout, " ")); << endl;
    // base.crossover(base);
    
    
}