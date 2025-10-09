# ALCF Hands-on HPC Workshop

[**2025 ALCF Hands-On HPC Workshop**](https://www.alcf.anl.gov/events/2025-alcf-hands-hpc-workshop)

The ALCF is hosting a Hands-on HPC Workshop which features both:

- Virtual event on **September 23--25** and an
- In-person event on **October 7--9, 2025**, at the TCS Conference Center at Argonne National Laboratory.

The workshop will provide an opportunity for hands-on time on Aurora, Polaris and a few of the AI Testbeds focusing on porting applications to heterogeneous architectures (CPU + GPU), improving code performance, and exploring AI/ML applications development on ALCF systems.

## Overview 

The repo is divided into sections following the schedule of both the virtual and in-person weeks:

**Virtual Week, Day 1** 
  * Hardware Overview of Aurora
  * Building, Compiling and Running Applications on Aurora
  * CPU and GPU affinity on Aurora
  * [Programming Models](programmingModels)

**Virtual Week, Day 2**
   * [Intel's Data Parallel Extensions for Python (DPEP)](data_science/intel_python)
   * [Learning Frameworks & Distributed Deep Learning on Aurora](data_science/AI_frameworks)
   * [Foundation Models on Aurora](data_science/AI_frameworks/ezpz)
   * [Workflows](workflows)
   * [Inference Service](inference_service)
   * [Agentic Workflows](agentic_workflows)

**Virtual Week, Day 3**
   * [Tools](tools)


## Obtaining the material and running the examples

Please clone the repo while on a Polaris or Aurora login node:

```
git clone --recurse-submodules https://github.com/argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
```

Then navigate to the particular material of interest and follow the instructions for executing the hands-on examples.

## Schedule

### Virtual week Sep. 23 - 25, 2025

#### Tuesday, Sept. 23

##### On Boarding on Aurora and Polaris

| Topic                                              | Duration | Time             | Speakers             |
| -------------------------------------------------- | -------- | ---------------- | -------------------- |
| Welcome Remarks                                    | 15 min   | 10:00-10:15 AM   | JaeHyuk Kwack, ANL   |
| Hardware Overview, Building, Compiling and Running | 90 min   | 10:15-11:45 AM   | Kris Rowe, ANL       |
| Affinity on Aurora                                 | 30 min   | 11:45 - 12:15 PM | Victor Anisimov, ANL |
| Break                                              | 30 min   | 12:15-12:45 PM   |                      |

##### Programming Models

| Topic  | Duration | Time            | Speakers                     |
| ------ | -------- | --------------- | ---------------------------- |
| SYCL   | 60 min   | 12:45 - 1:45 PM | Abhishek Baguestty, ANL      |
| OpenMP | 60 min   | 1:45 - 2:45 PM  | Colleen Bertoni, Ye Luo, ANL |
| Break  | 15 min   | 2:45 - 3:00 PM  |                              |
| KOKKOS | 60 min   | 3:00 - 4:00 PM  | Brian Homerding, ANL         |

#### Wednesday, Sept. 24

##### Overview of AI


| Topic                                                        | Duration | Time            | Speakers                            |
| ------------------------------------------------------------ | -------- | --------------- | ----------------------------------- |
| Intel Python                                                 | 30 min   | 10:00 - 10:30AM | Riccardo Balin, ANL                 |
| Torch                                                        | 50 min   | 10:30 - 11:20AM | Filippo Simini, Khalid Hossain, ANL |
| TensorFlow                                                   | 10 min   | 11:20 - 11:30AM | Filippo Simini, Khalid Hossain, ANL |
| JAX                                                          | 30 min   | 11:30 - 12:00AM | Filippo Simini, Khalid Hossain, ANL |
| Break                                                        | 30 min   | 12:00 - 12:30PM |                                     |
| [Foundation Models](https://samforeman.me/talks/2025/09/24/) | 45 min   | 12:30 -1:15 PM  | Sam Foreman, ANL                    |


##### WorkFlows and Data Management

| Topic             | Duration | Time           | Speakers                           |
| ----------------- | -------- | -------------- | ---------------------------------- |
| Workflows         | 60 min   | 1:15 - 2:15 PM | Christine Simpson, ANL             |
| Inference Service | 60 min   | 2:15 -3:15 PM  | Benoit Cote, Aditya Tanikanti, ANL |
| Agentic Workflows | 30 min   | 3:15 - 3:45 PM | Riccardo Balin, ANL                |

#### Thursday, Sept. 25

##### Tools Overview

| Topic                     | Duration | Time             | Speakers                |
| ------------------------- | -------- | ---------------- | ----------------------- |
| Linaro Tools              | 90 min   | 10:00 - 11:30 AM | Rudy Shand              |
| iprof                     | 30 min   | 11:30 - 12:00 PM | Thomas Applencourt, ANL |
| Break                     | 30 min   | 12:00 - 12:30 PM |                         |
| Intel Tools               | 90 min   | 12:30 - 2:00 PM  | JaeHyuk Kwack, ANL      |
| Overview of Visualization | 30 min   | 2:00 - 2:30 PM   | Joe Insley, ANL         |

### In-Person: Oct. 7 - Oct. 9

#### Tuesday, Oct. 7

##### Tools and Extended Hands-on Time

| Topic                                                                                                 | Duration | Time             | Speakers                             |
| ----------------------------------------------------------------------------------------------------- | -------- | ---------------- | ------------------------------------ |
| Welcome                                                                                               | 20 min   | 9:00 - 9:20 AM   | Yasaman Ghadar, ANL                  |
| HPCToolkit + Hands-On                                                                                 | 90 min   | 9:20 - 10:50 AM  | John Mellor-Crummey, Rice University |
| AM Break                                                                                              | 10 min   | 10:50 -11:00 AM  |                                      |
| Hands-On Time                                                                                         | 60 min   | 11:00 - 12:00 PM |                                      |
| Working Lunch; AI and HPC Applications on Leadership Computing Platforms: Performance and Scalability | 60 min   | 12:00 - 1:00 PM  | JaeHyuk Kwack, ANL                   |
| TAU + Hands-On                                                                                        | 90 min   | 1:00 - 2:30 PM   | Sameer Shende, University of Oregon  |
| PM Break                                                                                              | 15 min   | 2:30 - 2:45 PM   |                                      |
| Hands-On Time                                                                                         | 75 min   | 2:45 -4:00 PM    |                                      |
| Argonne Tour                                                                                          | 60 min   | 4:00 - 5:00 PM   |                                      |

#### Wednesday, Oct. 8

##### AI and Extended Hands-on Time

| Topic                                                                                                               | Duration | Time             | Speakers            |
| ------------------------------------------------------------------------------------------------------------------- | -------- | ---------------- | ------------------- |
| Profiling AI/ML                                                                                                     | 60 min   | 9:00 - 10:00 AM  | Khalid Hossain, ANL |
| Data Pipelines                                                                                                      | 60 min   | 10:00 - 11:00 AM | Huihuo Zheng, ANL   |
| AM Break                                                                                                            | 10 min   | 11:00 -11:10 AM  |                     |
| Hands-On Time                                                                                                       | 50 min   | 11:10 - 12:00 PM |                     |
| Group Photo                                                                                                         | 10 min   | 12:00 - 1:00PM   | Everyone            |
| [AERIS: Argonne Earth Systems Model for Reliable and Skillful Predictions](https://samforeman.me/talks/2025/10/08/) | 60 min   | 12:00 - 1:00 PM  | Sam Foreman, ANL    |
| AI Testbeds                                                                                                         | 60 min   | 1:00 - 2:00 PM   | Varuni Sastry, ANL  |
| PM Break                                                                                                            | 15 min   | 2:00 - 2:15 PM   |                     |
| Hands-On Time                                                                                                       | 175 min  | 2:15 -4:00 PM    |                     |
| Argonne Tour                                                           | 60 min   | 4:00 - 5:00 PM   |                     |


#### Thursday, Oct. 9

##### Data Management, I/O and Extended Hands-on Time

| Topic                                                                            | Duration | Time             | Speakers              |
| -------------------------------------------------------------------------------- | -------- | ---------------- | --------------------- |
| I/O Libraries (HDF5)                                                             | 60 min   | 9:00 - 10:00 AM  | Rob Latham, ANL       |
| DAOS and Checkpointing                                                           | 60 min   | 10:00 - 11:00 AM | Kaushik Velusamy, ANL |
| AM Break                                                                         | 10 min   | 11:00 -11:10 AM  |                       |
| Hands-On Time                                                                    | 50 min   | 11:00 - 12:00 PM |                       |
| Working Lunch, HACC I/O at Scale on Aurora: Early Experiences on Leveraging DAOS | 60 min   | 12:00 - 1:00 PM  | Steve Rangel, ANL     |
| Advanced DAOS                                                                    | 60 min   | 1:00 - 2:00 PM   | Kaushik Velusamy, ANL |
| PM Break                                                                         | 15 min   | 2:00 - 2:15 PM   |                       |
| Hands-On Time                                                                    | 45 min   | 2:15 -3:00 PM    |                       |
| Feedback and Adjourn                                                             | 15 min   | 3:00 - 3:15 PM   | Yasaman Ghadar, ANL   |

