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


#include <iostream>
#include <fstream>
#include <string>
#include "mimetik.h"
#include "multilayerPerceptron.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << " __  __ _                _   _ _    "      << endl;
    cout << "|  \\/  (_)              | | (_) |   "     << endl;
    cout << "| \\  / |_ _ __ ___   ___| |_ _| | __"     << endl;
    cout << "| |\\/| | | '_ ` _ \\ / _ \\ __| | |/ /"   << endl;
    cout << "| |  | | | | | | | |  __/ |_| |   < "      << endl;
    cout << "|_|  |_|_|_| |_| |_|\\___|\\__|_|_|\\_\\"  << endl;

    mimetik mk;

    // parse argument
    if (argc > 1)
    {
        string filename = argv[1];
        ifstream file;
        file.open(filename.c_str());
        if (!file.is_open())
        {
            cout <<  "error: can't open file " << filename << endl;
            return false;
        }

        mk.executeScript(filename);
        return 0;
    }

    while(true)
    {
        cout << endl << "Mimetik> ";
        string cmd;
        getline(cin, cmd);
        if (cmd == "exit")
            break;
        else
            mk.executeCommandLine(cmd);
    }
    return 0;
}
