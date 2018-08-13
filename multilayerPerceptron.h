/*
* MIT License
*
* Mimetik - machine learning software
* Copyright (c) 2018 Lounis Bellabes
* nolius@users.sourceforge.net
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <vector>
#include <iostream>
using namespace std;

struct traningSetMlp
{
    vector<double> tabExamples;
    vector<double> tabOutputTargets;
};

struct neuron
{
    double output;
    double error;
    vector<double> weight;
    vector<double> deltaWeight;
};

struct layer
{
public:
    vector<neuron> tabNeurons;
};

class multilayerPerceptron
{
public:
    multilayerPerceptron(const vector<int> tabNbNeurons, const double eta = 0.5, const double alpha = 0.9);
    ~multilayerPerceptron();
    bool loadTrainingSetFile(const string fileUrl);
    void setEta(const double eta);
    void setAlpha(const double alpha);
    bool loadTrainingSet(const vector< vector<double> > &tabInputs, const vector< vector<double> > &tabOutputTargets, const bool verbose = false);
    bool computeOutput(const vector<double> &tabInput, vector<double> &tabOutput);
    bool computeFile(const string fileInUrl, string fileOutUrl = "");
    bool learning(const int limit, const bool verbose = false, const bool randomShuffleTrainingSet = false);

    bool saveState(const string fileUrl);               // save neural network state (weights) in bin file
    bool loadState(const string fileUrl);               // load neural network state (weights) in bin file
    bool saveStateText(const string fileUrl);           // save neural network state (weights) in text file
    bool loadStateText(const string fileUrl);           // load neural network state (weights) in text file

private:
    double m_alpha;                                     // momentum factor [0,1]
    double m_eta;                                       // learning rate factor [0,1]
    vector<layer> m_neuralNetwork;                      // neural network
    vector<traningSetMlp> m_trainingSet;                // training set inputs and outputs
    void initLayers(const vector<int> &tabNbNeurons);
};

#endif // MULTILAYERPERCEPTRON_H
