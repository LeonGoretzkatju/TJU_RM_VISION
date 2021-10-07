#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "util.hpp"
#include "configurations.hpp"
#include "serial.hpp"
#include "DfcDetector.hpp"

using namespace std;
using namespace cv;
//Dfc id为4
class DfcHiter : public ModuleBase{
public:
    DfcHiter(SerialManager *_serial,DfcBaseDetetor* _dfcBaseDetetor):ModuleBase(4)
    {
        serial = _serial;
        dfcBaseDetector = _dfcBaseDetetor;
    }

protected:
    SerialManager *serial;
    DfcBaseDetetor *dfcBaseDetector;    
};

class InfancyDfcHiter : public DfcHiter
{
public:

    
    InfancyDfcHiter(SerialManager *_serial,DfcBaseDetetor *_dfcBaseDetetor) : DfcHiter(_serial,_dfcBaseDetetor)
    {

    }

      void Update(ImageData &frame,float dtTime)
    {   
        dfcBaseDetector->DetectDfcArmors(frame.image);
    }

protected:
    
};