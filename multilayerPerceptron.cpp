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

#include "multilayerPerceptron.h"
#include <fstream>
#include <string>
#include <math.h>
#include <time.h>
#include <algorithm>

multilayerPerceptron::multilayerPerceptron(const vector<int> tabNbNeurons, const double eta, const double alpha)
{
    if (tabNbNeurons.size() < 2)
    {
        cout <<  "Error: the neural network must contain at least 2 layers" << endl;
        return;
    }

    m_alpha = alpha;
    m_eta = eta;
    // create layers
    initLayers(tabNbNeurons);
}

multilayerPerceptron::~multilayerPerceptron()
{
}

bool multilayerPerceptron::loadTrainingSet(const vector< vector<double> > &tabInputs, const vector< vector<double> > &tabOutputTargets, const bool verbose)
{
    if (tabInputs.size() != tabOutputTargets.size())
    {
        cout <<  "Error: the number of inputs data does not match with the number of outputs data" << endl;
        return false;
    }

    for (int i=0; i < tabInputs.size(); i++)
    {
        if (tabInputs[i].size() != m_neuralNetwork[0].tabNeurons.size())
        {
            cout <<  "Error: the number of inputs data does not match" << endl;
            return false;
        }
    }

    for (int i=0; i < tabOutputTargets.size(); i++)
    {
        if (tabOutputTargets[i].size() != m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size())
        {
            cout <<  "Error: the number of outputs data does not match" << endl;
            return false;
        }
    }

    m_trainingSet.resize(tabInputs.size());
    for (int i=0; i < m_trainingSet.size(); i++)
    {
        m_trainingSet[i].tabExamples = tabInputs[i];
        m_trainingSet[i].tabOutputTargets = tabOutputTargets[i];
    }

    return true;
}

bool multilayerPerceptron::loadTrainingSetFile(const string fileUrl)
{
    ifstream file;
    file.open(fileUrl.c_str());

    if (!file.is_open())
    {
        cout <<  "Error: can't open file " << fileUrl << endl;
        return false;
    }

    string word;
    int nbInput = 0;
    int nbOutput = 0;
    int nbSample = 0;

    file >> word;
    if (word !=  "[mlp]")
    {
        cout <<  "Error can't find [mlp] tag in file: " << fileUrl << endl;
        file.close();
        return false;

    }

    file >> nbInput >> nbOutput >> nbSample;

    if (nbInput != m_neuralNetwork[0].tabNeurons.size())
    {
        cout <<  "Error: the number of inputs data does not match the number defined in file " << fileUrl << endl;
        file.close();
        return false;
    }
    else if (nbOutput != m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size())
    {
        cout <<  "Error: the number of outputs data does not match the number defined in file " << fileUrl << endl;
        file.close();
        return false;
    }

    file >> word;
    if (word !=  "[inputs]")
    {
        cout <<  "Error: can't find [inputs] tag in file " << fileUrl << endl;
        file.close();
        return false;
    }

    m_trainingSet.resize(nbSample);
    for (int i=0; i < m_trainingSet.size(); i++)
    {
        m_trainingSet[i].tabExamples.resize(nbInput);

        for (int j=0; j < m_trainingSet[i].tabExamples.size(); j++)
        {
            file >> m_trainingSet[i].tabExamples[j];
        }
    }

    file >> word;
    if (word !=  "[outputs]")
    {
        cout <<  "Error: can't find [outputs] tag in file " << fileUrl << endl;
        file.close();
        return false;
    }

    for (int i=0; i < m_trainingSet.size(); i++)
    {
        m_trainingSet[i].tabOutputTargets.resize(nbOutput);

        for (int j=0; j < m_trainingSet[i].tabOutputTargets.size(); j++)
        {
            file >> m_trainingSet[i].tabOutputTargets[j];
        }
    }

    file.close();
    return true;
}

void multilayerPerceptron::setEta(const double eta)
{
    m_eta = eta;
}

void multilayerPerceptron::setAlpha(const double alpha)
{
    m_alpha = alpha;
}

bool multilayerPerceptron::computeOutput(const vector<double> &tabInput, vector<double>& tabOutput)
{
    if (tabInput.size() != m_neuralNetwork[0].tabNeurons.size())
    {
        cout <<  "Error: the number of inputs data does not match" << endl;
        return false;
    }

    // set inputs
    for (int i=0; i < tabInput.size(); i++)
        m_neuralNetwork[0].tabNeurons[i].output = tabInput[i];

    // compute output
    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            double sum = 0;
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
                sum += m_neuralNetwork[i-1].tabNeurons[k].output * m_neuralNetwork[i].tabNeurons[j].weight[k];

            // sigmoid function
            m_neuralNetwork[i].tabNeurons[j].output = 1.0 / (1.0 + exp(-sum));
        }
    }

    // save outputs
    tabOutput.resize(m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size());
    for (int i=0; i < m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size(); i++)
        tabOutput[i] = m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons[i].output;

    return true;
}

bool multilayerPerceptron::computeFile(const string fileInUrl, string fileOutUrl)
{
    if (fileOutUrl == "")
        fileOutUrl = fileInUrl + "_out.txt";

    ifstream fileIn;
    fileIn.open(fileInUrl.c_str());

    if (!fileIn.is_open())
    {
        cout <<  "error: can't open file " << fileInUrl << endl;
        return false;
    }

    string word;
    int nbInput = 0;
    int nbOutput = 0;
    int nbSample = 0;

    fileIn >> word;
    if (word !=  "[mlp]")
    {
        cout <<  "Error can't find [mlp] tag in file: " << fileInUrl << endl;
        fileIn.close();
        return false;
    }

    fileIn >> nbInput >> nbOutput >> nbSample;

    if (nbInput != m_neuralNetwork[0].tabNeurons.size())
    {
        cout <<  "Error: the number of inputs data does not match the number defined in file " << fileInUrl << endl;
        fileIn.close();
        return false;
    }
    else if (nbOutput != m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size())
    {
        cout <<  "Error: the number of outputs data does not match the number defined in file " << fileInUrl << endl;
        fileIn.close();
        return false;
    }

    fileIn >> word;
    if (word !=  "[inputs]")
    {
        cout <<  "Error: can't find [inputs] tag in file " << fileInUrl << endl;
        fileIn.close();
        return false;
    }

    vector< vector<double> > tabSample;
    tabSample.resize(nbSample);
    for (int i=0; i < tabSample.size(); i++)
    {
        tabSample[i].resize(nbInput);

        for (int j=0; j < tabSample[i].size(); j++)
        {
            fileIn >> tabSample[i][j];
        }
    }

    ofstream fileOut(fileOutUrl.c_str(), ios::out | ios::trunc);
    if (!fileOut.is_open())
    {
        cout <<  "Error: can't open file " << fileOutUrl << endl;
        fileIn.close();
        return false;
    }

    fileOut << "[mlp_result]" << endl;

    // compute and save all samples
    for (int i=0; i < tabSample.size(); i++)
    {
        vector<double> tabOutput;
        computeOutput(tabSample[i], tabOutput);

        // write input
        fileOut << "Input: ";
        for (int j=0; j < tabSample[i].size(); j++)
            fileOut << tabSample[i][j] << " ";

        // write output
        fileOut << endl << "Output: ";
        for (int k=0; k < tabOutput.size(); k++)
            fileOut << tabOutput[k] << " ";

        fileOut << endl << endl;
    }

    fileIn.close();
    fileOut.close();
    return true;
}

bool multilayerPerceptron::learning(const int limit, const bool verbose, const bool randomShuffleTrainingSet)
{
    // check training set
    if (m_trainingSet.size() < 1)
    {
        cout <<  "Error: no training set loaded" << endl;
        return false;
    }
    else if (m_trainingSet[0].tabExamples.size() != m_neuralNetwork[0].tabNeurons.size())
    {
        cout <<  "Error: the number of inputs of the training set does not match with the neural network layers" << endl;
        return false;
    }
    else if (m_trainingSet[0].tabOutputTargets.size() != m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size())
    {
        cout <<  "Error: the number of outputs of the training set does not match with the neural network layers" << endl;
        return false;
    }

    srand((unsigned int) time(NULL));
    bool continueLearning = true;
    int nbLearning = 1;

    // set random weights
    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
            {
                m_neuralNetwork[i].tabNeurons[j].deltaWeight[k] = 0;
                m_neuralNetwork[i].tabNeurons[j].weight[k] = ((double) rand() / RAND_MAX) - 0.5;    //random value [-0.5; 0.5]
            }
        }
    }

    while (continueLearning)
    {
        // random training data order (sometimes, gives better results)
        if (randomShuffleTrainingSet)
            random_shuffle(m_trainingSet.begin(), m_trainingSet.end());

        // learn all training patterns
        double learningError = 0;
        for(int np=0; np < m_trainingSet.size(); np++)
        {
            // set inputs
            for (int i=0; i < m_trainingSet[np].tabExamples.size(); i++)
                m_neuralNetwork[0].tabNeurons[i].output = m_trainingSet[np].tabExamples[i];

            // compute output
            for(int i=1; i < m_neuralNetwork.size(); i++)
            {
                for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
                {
                    double sum = 0;
                    for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
                        sum += m_neuralNetwork[i-1].tabNeurons[k].output * m_neuralNetwork[i].tabNeurons[j].weight[k];

                    // sigmoid function
                    m_neuralNetwork[i].tabNeurons[j].output = 1.0 / (1.0 + exp(-sum));
                }
            }

            double RmsError = 0;
            for(int i=0; i < m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size(); i++)
            {
                double output = m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons[i].output;
                double target = m_trainingSet[np].tabOutputTargets[i];
                m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons[i].error = (target - output) * output * (1.0 - output);

                RmsError += pow((target - output), 2);
            }
            RmsError = sqrt(RmsError / (double) m_neuralNetwork[m_neuralNetwork.size()-1].tabNeurons.size());

            // error backpropagation
            for(int i = m_neuralNetwork.size()-2; i >= 0; i--)
            {
                for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
                {
                    double output = m_neuralNetwork[i].tabNeurons[j].output;
                    double sum = 0;
                    for (int k=0; k < m_neuralNetwork[i+1].tabNeurons.size(); k++)
                        sum += m_neuralNetwork[i+1].tabNeurons[k].weight[j] * m_neuralNetwork[i+1].tabNeurons[k].error;
                    m_neuralNetwork[i].tabNeurons[j].error = sum * output * (1.0 - output);
                }
            }

            // compute weights
            for(int i=1; i < m_neuralNetwork.size(); i++)
            {
                for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
                {
                    double error = m_neuralNetwork[i].tabNeurons[j].error;
                    for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
                    {
                        double deltaWeight = m_neuralNetwork[i].tabNeurons[j].deltaWeight[k];
                        double output = m_neuralNetwork[i-1].tabNeurons[k].output;
                        m_neuralNetwork[i].tabNeurons[j].deltaWeight[k] = m_eta * (output * error);
                        m_neuralNetwork[i].tabNeurons[j].weight[k] += m_neuralNetwork[i].tabNeurons[j].deltaWeight[k] + (m_alpha * deltaWeight);
                    }
                }
            }
            learningError = learningError + RmsError;
        }
        learningError = learningError / m_trainingSet.size();

        if(limit > 0 && nbLearning >= limit)
            continueLearning = false;

        if (verbose)
            cout << "Epoch = " << nbLearning << " : " << "RMS Error = "  << learningError << endl;
        nbLearning++;
    }
    return true;
}

bool multilayerPerceptron::saveState(const string fileUrl)
{
    ofstream file(fileUrl.c_str(), ios::out | ios::binary);
    if (!file.is_open())
    {
        cout <<  "error: can't open file " << fileUrl << endl;
        return false;
    }

    // save neural network structure: save nb of layers
    int nbLayer = m_neuralNetwork.size();
    file.write((char *) &nbLayer, sizeof nbLayer);

    // save neural network structure: save nb of neurons per layer
    for (int i=0; i < nbLayer; i++)
    {
        int nbNeuron = m_neuralNetwork[i].tabNeurons.size();
        file.write((char *) &nbNeuron, sizeof nbNeuron);
    }

    // save weights
    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
            {
                double weight = m_neuralNetwork[i].tabNeurons[j].weight[k];
                file.write((char *) &weight, sizeof weight);
            }
        }
    }

    file.close();
    return true;
}

bool multilayerPerceptron::loadState(const string fileUrl)
{
    ifstream file;
    file.open(fileUrl.c_str(), ios::in | ios::binary);
    if (!file.is_open())
        return false;

    // load neural network structure: load nb of layers
    int nbLayer = 0;
    file.read((char *) &nbLayer, sizeof nbLayer);

    // load neural network structure: load nb of neurons per layer
    vector<int> tabNbNeurons;
    int nbNeuron = 0;
    for (int i=0; i < nbLayer; i++)
    {
        file.read((char *) &nbNeuron, sizeof nbNeuron);
        tabNbNeurons.push_back(nbNeuron);
    }

    // adapt layers
    initLayers(tabNbNeurons);

    // load weights
    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
            {
                double weight;
                file.read((char *) &weight, sizeof weight);
                m_neuralNetwork[i].tabNeurons[j].weight[k] = weight;
            }
        }
    }

    file.close();
    return true;
}

bool multilayerPerceptron::saveStateText(const string fileUrl)
{
    ofstream file(fileUrl.c_str(), ios::out | ios::trunc);
    if (!file.is_open())
    {
        cout <<  "error: can't open file " << fileUrl << endl;
        return false;
    }

    file << "[mlp_layers]" << endl;

    // save neural network structure: save nb of layers
    int nbLayer = m_neuralNetwork.size();
    file << nbLayer;

    // save neural network structure: save nb of neurons per layer
    for (int i=0; i < nbLayer; i++)
    {
        int nbNeuron = m_neuralNetwork[i].tabNeurons.size();
        file << " " << nbNeuron ;
    }

    file << endl << endl << "[mlp_weights]" << endl;
    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
                file << m_neuralNetwork[i].tabNeurons[j].weight[k] << " ";

            file << endl ;
        }
        file << endl ;
    }

    file.close();
    return true;
}

bool multilayerPerceptron::loadStateText(const string fileUrl)
{
    ifstream file;
    file.open(fileUrl.c_str());
    if (!file.is_open())
        return false;

    string word;
    file >> word;
    if (word !=  "[mlp_layers]")
    {
        file.close();
        return false;
    }

    // load neural network structure: load nb of layers
    int nbLayer = 0;
    file >> nbLayer;

    // load neural network structure: load nb of neurons per layer
    vector<int> tabNbNeurons;
    int nbNeuron = 0;
    for (int i=0; i < nbLayer; i++)
    {
        file >> nbNeuron;
        tabNbNeurons.push_back(nbNeuron);
    }

    // adapt layers
    initLayers(tabNbNeurons);

    file >> word;
    if (word !=  "[mlp_weights]")
        return false;

    for(int i=1; i < m_neuralNetwork.size(); i++)
    {
        for(int j=0; j < m_neuralNetwork[i].tabNeurons.size(); j++)
        {
            for (int k=0; k < m_neuralNetwork[i-1].tabNeurons.size(); k++)
                file >> m_neuralNetwork[i].tabNeurons[j].weight[k];
        }
    }

    file.close();
    return true;
}

void multilayerPerceptron::initLayers(const vector<int> &tabNbNeurons)
{
    m_neuralNetwork.resize(tabNbNeurons.size());
    for (int i=0; i < tabNbNeurons.size(); i++)
    {
        m_neuralNetwork[i].tabNeurons.resize(tabNbNeurons[i]);
        for(int j=0; j < tabNbNeurons[i]; j++)
        {
            m_neuralNetwork[i].tabNeurons[j].output = 0;
            m_neuralNetwork[i].tabNeurons[j].error = 0;

            if(i == 0)
            {
                m_neuralNetwork[i].tabNeurons[j].weight.resize(0);
                m_neuralNetwork[i].tabNeurons[j].deltaWeight.resize(0);
            }
            else
            {
                m_neuralNetwork[i].tabNeurons[j].weight.resize(tabNbNeurons[i-1]);
                m_neuralNetwork[i].tabNeurons[j].deltaWeight.resize(tabNbNeurons[i-1]);
            }
        }
    }
}
