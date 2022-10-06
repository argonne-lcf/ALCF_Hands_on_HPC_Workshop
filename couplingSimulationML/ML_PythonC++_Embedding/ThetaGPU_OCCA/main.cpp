#include <math.h>
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>

#include <occa.hpp>

//---[ Internal Tools ]-----------------
// Note: These headers are not officially supported
//       Please don't rely on it outside of the occa examples
#include <occa/internal/utils/cli.hpp>
#include <occa/internal/utils/testing.hpp>
//======================================

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

constexpr int NX=256; // number of points in spatial discretization
occa::json parseArgs(int argc, const char **argv);
void PyIt(PyObject *p_func, double *u);
void pynalyze(PyObject *p_func);

int main(int argc, const char **argv) {

  //****Some python initialization***
  Py_Initialize();
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append(\".\")");

     std::cout << "Initialization of Python: Done" << std::endl;

  // initialize numpy array library
  import_array1(-1);

  // Load Python Module File
  PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName); // finished with this string so release reference
     std::cout << "Loaded Python Module File: Done" << std::endl;

  PyObject* py_PlotField = PyObject_GetAttrString(pModule, "analyses_plotField");
  PyObject* py_SVD = PyObject_GetAttrString(pModule, "analyses_SVD");
  PyObject* pcollect = PyObject_GetAttrString(pModule, "collection_func");
  Py_DECREF(pModule); // finished with this module so release reference
     std::cout << "Loaded Functions: Done" << std::endl;

  occa::json args = parseArgs(argc, argv);

  const double PI = 3.1415926535;	
  const double h = 2.0*PI/NX;
  const double cfl = 0.05;
  const double dt = 0.001; // 
  const double FT = 2.000; // Final Time
  const double NU = 0.01;  // diffusion param

  double s1 = dt / (2.0*h);
  double s2 = dt*NU / (h*h);

  double *uh      = new double[NX+2];
  double *uh_prev = new double[NX+2];
  double *res_par = new double[NX+2];

  double x;
  //Initialize
  for (int i = 1; i < NX+1; ++i) {
    x    = (double) 2.0*(i-1)*PI/NX;
    uh[i]      = sin(x);
    uh_prev[i] = sin(x);
  }
  uh[0]   = uh[NX]; // Ghost Nodes
  uh[NX+1] = uh[1]; // Ghost Nodes
  
  uh_prev[0]    = uh_prev[NX]; // Ghost Nodes
  uh_prev[NX+1] = uh_prev[1]; // Ghost Nodes

  occa::device device;
  occa::memory o_uh, o_uh_prev;

  //---[ Device Setup ]-------------------------------------
  device.setup((std::string) args["options/device"]);
  device.setup({
     {"mode"     , "CUDA"},
     {"device_id", 0},
   });

  // Allocate memory on the device
  o_uh = device.malloc<double>(NX+2);
  o_uh_prev = device.malloc<double>(NX+2);
  
  //Get Backend Pointer
  double *d_b = static_cast<double *>(o_uh.ptr());

  // Compile the kernel at run-time
  occa::kernel burgerUpdateKernel = device.buildKernel("kernel/burger.okl","update_burger");

  // Copy memory to the device
  o_uh_prev.copyFrom(uh_prev);
  double t = 0.0;
  int Ntri = NX+2;
  int i=0;
  auto walltime_start = std::chrono::high_resolution_clock::now();
  do{
      //Do the Burger's update with FD  
      burgerUpdateKernel(Ntri, s1, s2, o_uh_prev, o_uh);

      {
        PyIt(pcollect, d_b); //collect to global python data array  
      }

      // Move the current solution to the previous timestep 
      o_uh_prev.copyFrom(o_uh);
      t = t + dt;  
      if (i % 100 == 0) std::cout << "time = " << t << std::endl;
      i = i + 1;

  }while(t<FT);
  auto walltime_finish = std::chrono::high_resolution_clock::now();
  double wallTime = std::chrono::duration<double,std::milli>(walltime_finish-walltime_start).count(); 
  std::cout << "Mean Wall-Time: " << wallTime/i << std::endl;
  
  //Copy Result to Host
  o_uh.copyTo(res_par);
  std::cout << "A random value in the solution array: " << res_par[10] << std::endl;

      {
         Py_DECREF(pcollect);
      }

      //Plot the field
      {
	pynalyze(py_PlotField);
        Py_DECREF(py_PlotField);
      }	
      
      //SVD
      {
	pynalyze(py_SVD);
        Py_DECREF(py_SVD);
      }

      return 0;
}

occa::json parseArgs(int argc, const char **argv) {
  occa::cli::parser parser;
  parser
    .withDescription(
      "Example adding two vectors"
    )
    .addOption(
      occa::cli::option('d', "device",
                        "Device properties (default: \"{mode: 'Serial'}\")")
      .withArg()
      .withDefaultValue("{mode: 'Serial'}")
    )
    .addOption(
      occa::cli::option('v', "verbose",
                        "Compile kernels in verbose mode")
    );

  occa::json args = parser.parseArgs(argc, argv);
  occa::settings()["kernel/verbose"] = args["options/verbose"];

  return args;
}

void PyIt(PyObject *p_func, double *u)
{
    PyObject* pArgs = PyTuple_New(1);

    //Numpy array dimensions
    npy_intp dim[] = {NX+2};

    // create a new Python array that is a wrapper around u (not a copy) and put it in tuple pArgs
    PyObject* array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
    PyTuple_SetItem(pArgs, 0, array_1d);

    // pass array into our Python function and cast result to PyArrayObject
    PyArrayObject* pValue = (PyArrayObject*) PyObject_CallObject(p_func, pArgs);
    std::cout <<"Called python data collection function successfully"<<std::endl;

    Py_DECREF(pArgs);
    Py_DECREF(pValue);
    // We don't need to decref array_1d because PyTuple_SetItem steals a reference
}

void pynalyze(PyObject *p_func)
{
  // panalsyses_func doesn't require an argument so pass nullptr
  PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(p_func, nullptr);
  std::cout << "Called python analyses function successfully"<<std::endl;

  Py_DECREF(pValue);
}
