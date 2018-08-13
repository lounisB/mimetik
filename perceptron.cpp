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

#include "perceptron.h"
#include <fstream>
#include <string>

perceptron::perceptron(const int nbEntry)
{
    m_tabWeights.resize(nbEntry);
}

perceptron::~perceptron()
{
}

bool perceptron::loadTrainingSetFile(const string fileUrl)
{
    ifstream file;
    file.open(fileUrl);

    if (!file.is_open())
    {
        cout <<  "Error: can't open file " << fileUrl << endl;
        return false;
    }

    string word;
    int nbInput = 0;
    int nbExample = 0;

    file >> word;
    if (word !=  "[perceptron]")
    {
        cout <<  "Error: can't find [perceptron] tag in file " << fileUrl << endl;
        file.close();
        return false;
    }

    file >> nbInput >> nbExample;
    file >> word;
    if (word !=  "[inputs]")
    {
        cout <<  "Error: can't find [inputs] tag in file " << fileUrl << endl;
        file.close();
        return false;
    }

    m_trainingSet.resize(nbExample);
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
        file >> m_trainingSet[i].outputTarget;
    }

    file.close();
    return true;
}

bool perceptron::loadTrainingSet(const vector< vector<double> > &tabInputs, vector<int>& tabOutputTargets, const bool verbose)
{
    if (tabInputs.size() != tabOutputTargets.size())
    {
        cout <<  "Error: the number of inputs data does not match with the number of outputs data" << endl;
        return false;
    }

    m_trainingSet.resize(tabInputs.size());
    for (int i=0; i < m_trainingSet.size(); i++)
    {
        m_trainingSet[i].tabExamples = tabInputs[i];
        m_trainingSet[i].outputTarget = tabOutputTargets[i];
    }

    if(verbose)
    {
        cout << "Examples :" << endl << endl;
        for (int i=0; i < m_trainingSet.size(); i++)
        {
            for (int j=0; j < m_trainingSet[i].tabExamples.size(); j++)
            {
                cout << m_trainingSet[i].tabExamples[j];
            }
            cout << endl;
        }

        cout << endl << "Outputs :" <<endl << endl;
        for (int i=0; i < m_trainingSet.size(); i++)
        {
            cout << m_trainingSet[i].outputTarget;
            cout << endl;
        }
    }
    return true;
}

void perceptron::computeOutput(const vector<double> &tabInput, int &output)
{
    int sum = 0;
    for (int i=0; i < m_tabWeights.size(); i++)
    {
        sum += tabInput[i] * m_tabWeights[i];
    }

    if (sum > 0)
        output = 1;
    else
        output = 0;
}

void perceptron::learning(const int limit, const bool verbose)
{
    int nbLearning = 1;
    bool continueLearning = true;
    double *saveTabWeights = new double[m_tabWeights.size()];
    for (int i=0; i < m_tabWeights.size(); i++)
    {
        saveTabWeights[i] = m_tabWeights[i];    // =0
    }

    while (continueLearning)
    {
        for (int i=0; i < m_trainingSet.size(); i++)
        {
            int output = 0;
            computeOutput(m_trainingSet[i].tabExamples, output);

            for (int j=0; j < m_tabWeights.size(); j++)
            {
                m_tabWeights[j] = m_tabWeights[j] + ( ((m_trainingSet[i].outputTarget - output) * m_trainingSet[i].tabExamples[j]) );
            }
        }

        if (verbose)
        {
            cout << endl << "Saved Weights / Weights :" <<endl << endl;
            for (int aff=0; aff < m_tabWeights.size(); aff++)
            {
                cout << saveTabWeights[aff] << "\t" << m_tabWeights[aff];
                cout << endl;
            }
        }

        bool difference = false;
        for (int i=0; i < m_tabWeights.size(); i++)
        {
            if (m_tabWeights[i] != saveTabWeights[i])
                difference = true;
        }

        if (!difference)
            continueLearning = false;

        if (limit > 0)
        {
            if(nbLearning >= limit)
                continueLearning = false;
        }

        for (int i=0; i < m_tabWeights.size(); i++)
        {
            saveTabWeights[i] = m_tabWeights[i];
        }
        nbLearning++;
    }

    if (verbose)
    {
        cout << endl << "Outputs :" << endl << endl;
        for (int i=0; i < m_trainingSet.size(); i++)
        {
            int Output = 0;
            computeOutput(m_trainingSet[i].tabExamples, Output);
            cout << Output << endl;
        }
    }
    delete saveTabWeights;
}

bool perceptron::saveStateText(const string fileUrl)
{
    ofstream file(fileUrl, ios::out | ios::trunc);

    if (!file.is_open())
        return false;

    file << "[weights]" << endl;
    for (int i=0; i < m_tabWeights.size(); i++)
    {
        file << m_tabWeights[i];
        file << endl;
    }

    file.close();
    return true;
}

bool perceptron::loadStateText(const string fileUrl)
{
    ifstream file;
    file.open(fileUrl);

    if (!file.is_open())
        return false;

    string word;
    file >> word;
    if (word !=  "[weights]")
        return false;

    for (int i=0; i < m_tabWeights.size(); i++)
    {
        file >> m_tabWeights[i];
    }

    file.close();
    return true;
}

