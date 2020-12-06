// cmake .. // From build
// make
#include <iostream>
#include <time.h>
#include <math.h>
#include <string.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

constexpr double PI = 3.1415926535;
constexpr double NU = 0.01; // parameter for PDE
constexpr int NX = 256; // number of points in spatial discretization
constexpr double DT = 0.001; // time step delta t
constexpr double FT = 2.0; // final time

void collect_data(PyObject *pcollection_func, double *u);
void analyse_data(PyObject *panalyses_func, double *u);
void initialize(double *u);
void update_solution(double *u, double *u_prev);

int main(int argc, char *argv[])
{
    // Some python initialization
    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    std::cout << "Initializing numpy library" << std::endl;
    // initialize numpy array library
    import_array1(-1);
    
    std::cout << "Loading python module" << std::endl;
    PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName); // finished with this string so release reference
    std::cout << "Loaded python module" << std::endl; 

    std::cout << "Loading functions from module" << std::endl;
    PyObject* pcollection_func = PyObject_GetAttrString(pModule, "collection_func");
    PyObject* panalyses_func = PyObject_GetAttrString(pModule, "analyses_func");
    Py_DECREF(pModule); // finished with this module so release reference
    std::cout << "Loaded functions" << std::endl;

    // Initialize array for the solution field u with the initial condition for the PDE
    double u[NX+2]; // length is number of spatial points (NX) plus 2 ghost points to handle boundary conditions
    initialize(u);

    double u_prev[NX+2]; // again include 2 ghost points
    initialize(u_prev); // intialized to same as u

    // Time loop for evolution of the Burgers equation
    clock_t start, end;
    double t, cpu_time_used;

    // Returns the _processor_ time consumed by the program
    start = clock();
    // Solve the problem
    t = 0.0;
    do{
      // solve PDE at next time step 
      update_solution(u,u_prev);

      // Exchanging data with python
      collect_data(pcollection_func,u);

      std::cout << "time = " << t << std::endl;;
      t = t + DT;
    }while(t<FT);

    Py_DECREF(pcollection_func);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    std::cout<<"CPU time = "<< cpu_time_used << std::endl;
    
    std::cout<<"Python based data analysis starting:"<< std::endl;
    
    analyse_data(panalyses_func,u);
    Py_DECREF(panalyses_func);

    return 0;
}

void initialize(double *u)
{
  double x;

  for (int i = 1; i < NX+1; i++)
  {
      x = (double) (i-1)/(NX) * 2.0 * PI;
      u[i] = sin(x);
  }

  // Handle the ghost points: periodic boundary conditions
  u[0] = u[NX];
  u[NX+1] = u[1];
}

void update_solution(double *u, double *u_prev)
{
  double dx = 2.0 * PI/NX; // delta x (spatial discretization)

  // loop over the array, updating solution u with a finite difference method
  // (based on values at the previous time in the neighborhood)
  // Burgers' equation: u_t + u*u_x = nu * u_xx
  // skips updating ghost points, one on either end
  for (int i = 1; i < NX+1; i++)
  {
      u[i] = u_prev[i] + NU*DT/(dx*dx)*(u_prev[i+1]+u_prev[i-1]-2.0*u_prev[i]) - DT/(2*dx)*(u_prev[i+1]-u_prev[i-1])*u_prev[i];
  } 

  // Handle the ghost points with periodic BCs
  u[0] = u[NX];
  u[NX+1] = u[1];

  // copy u into u_prev for use next time
  for (int i = 0; i < NX+2; i++)
  {
      u_prev[i] = u[i];
  }

}

void collect_data(PyObject *pcollection_func, double *u)
{
  PyObject* pArgs = PyTuple_New(1);
  
  //Numpy array dimensions
  npy_intp dim[] = {NX+2};

  // create a new Python array that is a wrapper around u (not a copy) and put it in tuple pArgs
  PyObject* array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);

  // pass array into our Python function and cast result to PyArrayObject
  PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(pcollection_func, pArgs); 
  std::cout << "Called python data collection function successfully"<<std::endl;

  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  // We don't need to decref array_1d because PyTuple_SetItem steals a reference 
}

void analyse_data(PyObject *panalyses_func, double *u)
{
  // panalsyses_func doesn't require an argument so pass nullptr 
  PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(panalyses_func, nullptr);
  std::cout << "Called python analyses function successfully"<<std::endl;

  // Printing out values of the SVD eigenvectors of the first and second modes for each field DOF
  for (int i = 0; i < 10; ++i) 
  {
    double* current = (double*) PyArray_GETPTR2(pValue, 0, i); // row 0, column i
    std::cout << "First mode value: " << *current << std::endl;
  }

  for (int i = 0; i < 10; ++i)
  {
    double* current = (double*) PyArray_GETPTR2(pValue, 1, i); // row 1, column i
    std::cout << "Second mode value: " << *current << std::endl;
  }

  Py_DECREF(pValue);
}
