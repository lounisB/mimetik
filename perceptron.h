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

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <iostream>
using namespace std;

struct traningSetP
{
    vector<double> tabExamples;
    int outputTarget;
};

class perceptron
{
public:
    perceptron(const int nbEntry);
    ~perceptron();
    bool loadTrainingSetFile(const string fileUrl);
    bool loadTrainingSet(const vector< vector<double> > &tabInputs, vector<int> &tabOutputTargets, const bool verbose = false);
    void computeOutput(const vector<double> &tabInput, int &output);
    void learning(const int limit = -1, const bool verbose = false);

    bool saveStateText(const string fileUrl);   // save neural network state (weights) in text file
    bool loadStateText(const string fileUrl);   // load neural network state (weights) in text file

private:
    vector<double> m_tabWeights;                // weights
    vector<traningSetP> m_trainingSet;          // training set inputs and outputs
};

#endif // PERCEPTRON_H
