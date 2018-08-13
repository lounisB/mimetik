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

#include "mimetik.h"
#include <fstream>
#include <string>
#include <cstdlib>

mimetik::mimetik()
{  
    vector<int> tabNbLayers;
    tabNbLayers.resize(3);
    tabNbLayers[0] = 1;
    tabNbLayers[1] = 10;
    tabNbLayers[2] = 1;
    m_mlp = new multilayerPerceptron(tabNbLayers);
}

mimetik::~mimetik()
{
    delete m_mlp;
}

bool mimetik::executeCommandLine(string cmd)
{
    // split cmd
    m_tabCmd.clear();
    size_t pos = 0;
    string delimiter = " ";
    string token;
    while ((pos = cmd.find(delimiter)) != string::npos)
    {
        token = cmd.substr(0, pos);
        if (token != "")
            m_tabCmd.push_back(token);
        cmd.erase(0, pos + delimiter.length());
    }
    if (cmd != "")
        m_tabCmd.push_back(cmd); // add last token

    if (m_tabCmd.size() < 1)
        return false;

    // command line interpreter
    bool ret = false;
    if (m_tabCmd[0] == "help")
        doHelp();
    else if (m_tabCmd[0] == "network")
        ret = doNetwork();
    else if (m_tabCmd[0] == "loadTrainingSet")
        ret = doLoadTrainingSet();
    else if (m_tabCmd[0] == "setEta")
        ret = doSetEta();
    else if (m_tabCmd[0] == "setAlpha")
        ret = doSetAlpha();
    else if (m_tabCmd[0] == "learning")
        ret = doLearning();
    else if (m_tabCmd[0] == "compute")
        ret = doCompute();
    else if (m_tabCmd[0] == "computeFile")
        ret = doComputeFile();
    else if (m_tabCmd[0] == "saveState")
        ret = doSaveState();
    else if (m_tabCmd[0] == "saveStateText")
        ret = doSaveStateText();
    else if (m_tabCmd[0] == "loadState")
        ret = doLoadState();
    else if (m_tabCmd[0] == "loadStateText")
        ret = doLoadStateText();
    else if (m_tabCmd[0] == "execute")
        ret = doExecute();
    else
    {
        cout << "unknown command: " << m_tabCmd[0] <<  endl;
        return false;
    }
    return ret;
}

bool mimetik::doNetwork()
{
    if (m_tabCmd.size() < 3)
    {
        cout << "a network must contain at least 2 layers (ex: network 1 10 1)" << endl;
        return false;
    }

    vector<int> tabNbLayers;
    for (int i = 1; i < m_tabCmd.size(); i++)
    {
       int nblayer = atoi( m_tabCmd[i].c_str());
       if (nblayer < 1)
       {
           cout << "nblayer must be an integer > 1" << endl;
           return false;
       }
       tabNbLayers.push_back(nblayer);
    }

    m_mlp = new multilayerPerceptron(tabNbLayers);
    cout << "new multilayer perceptron: ";
    for (int i = 0; i < tabNbLayers.size(); i++)
        cout << tabNbLayers[i] << " ";
    cout << endl;
    return true;
}

bool mimetik::doLoadTrainingSet()
{
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: loadTrainingSet trainingset.txt" << endl;
        return false;
    }

    string fileName = m_tabCmd[1];
    bool ret = m_mlp->loadTrainingSetFile(fileName);
    if (ret)
        cout << "training set file: " << fileName << " loaded" << endl;

    return ret;
}

bool mimetik::doSetEta()
{
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: setEta eta" << endl;
        cout << "example: setEta 0.5" << endl;
        return false;
    }

    double eta = atof( m_tabCmd[1].c_str());
    m_mlp->setEta(eta);
    cout << "eta = "<< eta << endl;
    return true;
}

bool mimetik::doSetAlpha()
{
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: setAlpha eta" << endl;
        cout << "example: setAlpha 0.9" << endl;
        return false;
    }

    double alpha = atof( m_tabCmd[1].c_str());
    m_mlp->setAlpha(alpha);
    cout << "alpha = "<< alpha << endl;
    return true;
}

bool mimetik::doLearning()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: learning limit verbose(booleen) random(booleen)" << endl;
        cout << "example: learning 5000" << endl;
        cout << "example: learning 5000 true false" << endl;
        return false;
    }

    int limit = atoi( m_tabCmd[1].c_str());
    if (limit < 1)
    {
        cout << "limit must be an integer > 1" << endl;
        return false;
    }

    cout << "start learning..." << endl;
    if (m_tabCmd.size() ==  2)
        ret = m_mlp->learning(limit);
    else if (m_tabCmd.size() == 3)
    {
        bool verbose = true;
        if (m_tabCmd[2] == "false")
            verbose = false;
        ret = m_mlp->learning(limit, verbose);
    }
    else if (m_tabCmd.size() > 3)
    {
        bool verbose = true;
        if (m_tabCmd[2] == "false")
            verbose = false;

        bool ramdom = false;
        if (m_tabCmd[3] == "true")
            verbose = true;
        ret = m_mlp->learning(limit, verbose, ramdom);
    }

    if (ret)
        cout << "learning ok" << endl;
    return ret;
}

bool mimetik::doCompute()
{
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: compute input1 input2 ..." << endl;
        cout << "example: compute 0.125 1 0.5" << endl;
        return false;
    }

    vector<double> tabInputs;
    for (int i = 1; i < m_tabCmd.size(); i++)
    {
       double input = atof( m_tabCmd[i].c_str());
       tabInputs.push_back(input);
    }
    cout << "inputs: ";
    for (int i = 0; i < tabInputs.size(); i++)
        cout << tabInputs[i] << " ";
    cout << endl;

    vector<double> tabOutputs;
    bool ret = m_mlp->computeOutput(tabInputs, tabOutputs);
    if (ret)
    {
        cout << "outputs: ";
        for (int i = 0; i < tabOutputs.size(); i++)
            cout << tabOutputs[i] << " ";
        cout << endl;
    }
    return ret;
}

bool mimetik::doComputeFile()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: computeFile fileIn fileOut" << endl;
        cout << "example: computeFile fileIn.txt" << endl;
        cout << "example: computeFile fileIn.txt fileOut.txt" << endl;
        return false;
    }

    string fileIn =  m_tabCmd[1];
    if (m_tabCmd.size() ==  2)
    {
        ret = m_mlp->computeFile(fileIn);
    }
    else if (m_tabCmd.size() > 2)
    {
        string fileOut =  m_tabCmd[2];
        ret = m_mlp->computeFile(fileIn, fileOut);
    }

    if (ret)
        cout << "computeFile ok" << endl;

    return ret;
}

bool mimetik::doSaveState()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: saveState filename" << endl;
        return false;
    }

    string filename =  m_tabCmd[1];
    if (m_tabCmd.size() ==  2)
        ret = m_mlp->saveState(filename);
    if (ret)
        cout << "saveState ok" << endl;

    return ret;
}

bool mimetik::doSaveStateText()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: saveStateText filename.txt" << endl;
        return false;
    }

    string filename =  m_tabCmd[1];
    if (m_tabCmd.size() ==  2)
        ret = m_mlp->saveStateText(filename);
    if (ret)
        cout << "saveStateText ok" << endl;

    return ret;
}

bool mimetik::doLoadState()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: loadState filename" << endl;
        return false;
    }

    string filename =  m_tabCmd[1];
    if (m_tabCmd.size() ==  2)
        ret = m_mlp->loadState(filename);
    if (ret)
        cout << "loadState ok" << endl;

    return ret;
}

bool mimetik::doLoadStateText()
{
    bool ret = false;
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: loadStateText filename.txt" << endl;
        return false;
    }

    string filename =  m_tabCmd[1];
    if (m_tabCmd.size() ==  2)
        ret = m_mlp->loadStateText(filename);
    if (ret)
        cout << "loadStateText ok" << endl;

    return ret;
}

bool mimetik::doExecute()
{
    if (m_tabCmd.size() < 2)
    {
        cout << "usage: execute script.mimetik" << endl;
        return false;
    }

    string filename =  m_tabCmd[1];
    return executeScript(filename);
}

bool mimetik::executeScript(const string filename)
{
    ifstream file;
    file.open(filename.c_str());
    if (!file.is_open())
    {
        cout <<  "error: can't open file " << filename << endl;
        return false;
    }

    string command;
    while (getline(file, command))
    {
        bool test = executeCommandLine(command);
        if (!test)
        {
            file.close();
            return false;
        }
    }

    file.close();
    return true;
}

bool mimetik::doHelp()
{
    cout << "Mimetik by Lounis Bellabes (MIT License)" << endl;
    cout << "usage:" << endl;
    cout << "\t" << "network nbLayer1 nbLayer2 ... - Create neural network layers" << endl;
    cout << "\t" << "loadTrainingSet trainingset.txt - Load training set from file" << endl;
    cout << "\t" << "setEta eta - set learning rate factor [0,1] (default = 0.5)" << endl;
    cout << "\t" << "setAlpha alpha - set momentum factor [0,1] (default = 0.9)" << endl;
    cout << "\t" << "learning limit verbose(booleen) randomOrder(booleen) - Start learning" << endl;
    cout << "\t" << "compute input1 input2 ... - compute outputs" << endl;
    cout << "\t" << "computeFile fileIn fileOut - compute a file" << endl;
    cout << "\t" << "saveState filename - Save neural network state in binary file" << endl;
    cout << "\t" << "saveStateText filename.txt - Load neural network state in text file" << endl;
    cout << "\t" << "loadState filename - Load neural network state from binary file" << endl;
    cout << "\t" << "loadStateText filename.txt - Load neural network state from text file" << endl;
    cout << "\t" << "execute script.mimetik - Execute mimetik script" << endl;
    cout << "\t" << "exit - Quit the software" << endl << endl;
    cout << "examples:" << endl;
    cout << "\t" << "network 2 10 5 1" << endl;
    cout << "\t" << "loadTrainingSet trainingset.txt" << endl;
    cout << "\t" << "setEta 0.5" << endl;
    cout << "\t" << "setAlpha 0.9" << endl;
    cout << "\t" << "learning 5000 true false" << endl;
    cout << "\t" << "compute 0.5 0.1" << endl;
    cout << "\t" << "computeFile fileIn.txt" << endl;
    cout << "\t" << "saveState weights.bin" << endl;
    cout << "\t" << "saveStateText weights.txt" << endl;
    cout << "\t" << "loadState weights.bin" << endl;
    cout << "\t" << "loadStateText weights.txt" << endl;
    cout << "\t" << "execute script.mimetik" << endl;
    return true;
}
