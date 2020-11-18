// cmake .. // From build
// make
// export PATH=/home/rmlans/anaconda3/envs/tf2_env/bin:$PATH
// export LD_LIBRARY_PATH=/home/rmlans/anaconda3/envs/tf2_env/lib:$LD_LIBRARY_PATH

#include <iostream>
#include <time.h>
#include <math.h>
#include <string.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

double PI = 3.1415926535;
double NU = 0.01;
int NX = 256;
double DT = 0.001;
double FT = 2.0;

void collect_data(PyObject *pcollection_func, double *u);
void analyse_data(PyObject *panalyses_func, double *u);
void initialize(double *u);
void update_solution(double *u, double *u_temp);

void init_numpy() {
  import_array1();
}

int main(int argc, char *argv[])
{

    // Some python initialization
    // Pointers for loading python modules, function names
    PyObject *pName, *pModule, *pcollection_func, *panalyses_func;

    int some_int = 0;

    Py_Initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");

    std::cout << "Initializing numpy library" << std::endl;
    // initialize numpy array library
    init_numpy();
    
    std::cout << "Loading python module" << std::endl;
    pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    pModule = PyImport_Import(pName);
    std::cout << "Loaded python module" << std::endl; 

    std::cout << "Loading functions from module" << std::endl;
    pcollection_func = PyObject_GetAttrString(pModule, "collection_func");
    panalyses_func = PyObject_GetAttrString(pModule, "analyses_func");
    std::cout << "Loaded functions" << std::endl;

    // Do the array initialization business for the solution field
    double u[NX+2]; // 2 GHOST POINTS
    initialize(u);

    double u_temp[NX+2]; // 2 GHOST POINTS
    initialize(u_temp);

    // Time loop for evolution of the Burgers equation
    clock_t start, end;
    double t, cpu_time_used;

    start = clock();
    // Solve the problem
    t = 0.0;
    do{
      update_solution(u,u_temp);
      // Exchanging data with python
      collect_data(pcollection_func,u);
      std::cout << "time = " << t << std::endl;;
      t = t + DT;
    }while(t<FT);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    std::cout<<"CPU time = "<< cpu_time_used << std::endl;
    
    std::cout<<"Python based data analysis starting:"<< std::endl;
    
    analyse_data(panalyses_func,u);

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

  // Update periodic BCs
  u[0] = u[NX];
  u[NX+1] = u[1];
}

void update_solution(double *u, double *u_temp)
{
  double dx = 2.0 * PI/NX;

  for (int i = 1; i < NX+1; i++)
  {
      u[i] = u_temp[i] + NU*DT/(dx*dx)*(u_temp[i+1]+u_temp[i-1]-2.0*u_temp[i]) - DT/(2*dx)*(u_temp[i+1]-u_temp[i-1])*u_temp[i];
  }

  for (int i = 1; i < NX+1; i++)
  {
      u_temp[i] = u[i];
  }  

  // Update periodic BCs
  u[0] = u[NX];
  u[NX+1] = u[1];

  // Update periodic BCs
  u_temp[0] = u_temp[NX];
  u_temp[NX+1] = u_temp[1];

}

void collect_data(PyObject *pcollection_func, double *u)
{

  PyObject *pArgs, *array_1d;
  PyArrayObject *pValue;
  pArgs = PyTuple_New(1);
  
  //Numpy array dimensions
  npy_intp dim[] = {NX+2};
  // create a new array
  array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);
  pValue = (PyArrayObject*)PyObject_CallObject(pcollection_func, pArgs); //Casting to PyArrayObject
  std::cout << "Called python data collection function successfully"<<std::endl;

  Py_DECREF(pArgs);
  Py_DECREF(pValue);
  PyArray_ENABLEFLAGS((PyArrayObject*)array_1d, NPY_ARRAY_OWNDATA); // Deallocate array_1d
  // Py_DECREF(array_1d);
}

void analyse_data(PyObject *panalyses_func, double *u)
{

  PyObject *pArgs, *array_1d;
  PyArrayObject *pValue;
  pArgs = PyTuple_New(1);
  
  //Numpy array dimensions
  npy_intp dim[] = {NX+2};
  // create a new array
  array_1d = PyArray_SimpleNewFromData(1, dim, NPY_FLOAT64, u);
  PyTuple_SetItem(pArgs, 0, array_1d);
  pValue = (PyArrayObject*)PyObject_CallObject(panalyses_func, pArgs); //Casting to PyArrayObject
  std::cout << "Called python analyses function successfully"<<std::endl;

  Py_DECREF(pArgs);
  PyArray_ENABLEFLAGS((PyArrayObject*)array_1d, NPY_ARRAY_OWNDATA); // Deallocate array_1d
  // Py_DECREF(array_1d);

  // Printing out values of the SVD eigenvectors of the first and second modes for each field DOF
  double* c_out = reinterpret_cast<double*>(PyArray_DATA(pValue));
  for (int i = 0; i < 10; ++i) // Only printing 10 out of NX for checking the order of allocation in arrays
  {
    std::cout << "First mode value: " << (*(c_out+i)) << std::endl;
  }

  for (int i = 0; i < 10; ++i) // Only printing 10 out of NX for checking the order of allocation in arrays
  {
    std::cout << "Second mode value: " << (*(c_out+NX+i)) << std::endl;
  }

  Py_DECREF(pValue);

  // Null and delete the pointer you allocated to get data from python
  c_out = nullptr;
  delete[] c_out;
}