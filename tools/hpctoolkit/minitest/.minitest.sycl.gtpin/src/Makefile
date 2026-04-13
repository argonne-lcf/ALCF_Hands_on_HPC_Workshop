# SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
#
# SPDX-License-Identifier: Apache-2.0

TARGET = minitest.sycl

SRCDIR = .

SRCS = $(SRCDIR)/single.cc $(SRCDIR)/syclgpu.cc

CXX = $(HPCTOOLKIT_MINISYCL_MPI_CXX)

CPPFLAGS = -DUSE_MPI
CXXFLAGS = -g -O2 -fsycl

default: $(TARGET)

$(TARGET): $(SRCS)  $(SRCDIR)/maincommon.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(TARGET) $(SRCS)

clean veryclean distclean:
	rm -f $(TARGET)
